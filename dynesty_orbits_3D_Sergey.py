import os
import h5py
import pickle
import dynesty
import numpy as np
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool

import scipy
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm, truncnorm, chi2
from scipy.spatial.transform import Rotation
from sklearn.mixture import GaussianMixture

import astropy.units as u
from astropy.constants import G
G = G.to(u.pc * u.Msun**-1 * (u.km / u.s)**2)

import gala.potential as gp
import gala.dynamics as gd
import gala.integrate as gi
import gala.units as gu

from gala.units import galactic

BAD_VAL = -1e-50

### Prior transform function ###
def prior_transform(utheta):
    # Unpack the unit cube values
    u_logM, u_Rs, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, \
    u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, \
    u_a, u_b, u_c, u_kx, u_ky, u_kz = utheta

    logM_min, logM_max     = 11, 13
    Rs_min, Rs_max         = 5, 25
    
    x_pos_min, x_pos_max = -75, -25
    y_pos_min, y_pos_max = -5, 5
    z_pos_min, z_pos_max = -75, 75

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  

    pos_init_x = x_pos_min + u_pos_init_x * (x_pos_max - x_pos_min) 
    pos_init_y = y_pos_min + u_pos_init_y * (y_pos_max - y_pos_min) 
    pos_init_z = z_pos_min + u_pos_init_z * (z_pos_max - z_pos_min) 

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = abs( norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel) )
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    a = norm.ppf(u_a, loc=0, scale=1)
    b = norm.ppf(u_b, loc=0, scale=1)
    c = norm.ppf(u_c, loc=0, scale=1)

    df = 3
    kx = chi2.ppf(u_kx, df-(0+1)+1)**0.5
    ky = chi2.ppf(u_ky, df-(1+1)+1)**0.5
    kz = chi2.ppf(u_kz, df-(2+1)+1)**0.5

    # Return the transformed parameters
    return (logM, Rs,
            pos_init_x, pos_init_y, pos_init_z, 
            vel_init_x, vel_init_y, vel_init_z,
            t_end, 
            a, b, c, kx, ky, kz)

def log_likelihood_GMM(params, dict_data):

    # Unpack the data
    x_data = dict_data['x']
    y_data = dict_data['y']
    sigma  = dict_data['sigma']

    # Generate model predictions for the given parameters
    x_model, y_model = model(params, type='model')

    gmm = GaussianMixtureModel(x_model, y_model, sigma)

    samples = np.concatenate((x_data[:, None], y_data[:, None]), axis=1)
    logl    = np.sum( gmm.score_samples(samples) )

    return logl

def model(params, type='data'):
    # Unpack parameters
    logM, Rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end, \
    a, b, c, kx, ky, kz = params

    # Wishart Prior
    df    = 3
    scale = np.identity(df)

    my_wishart        = MyWishart(a,b,c,kx,ky,kz)
    covariance_matrix = my_wishart.rvs(df, scale)
    eigvals, eigvec   = scipy.linalg.eigh(covariance_matrix)

    q1, q2, q3 = eigvals**0.5
    rot_mat    = eigvec

    pot_NFW = gp.NFWPotential(10**logM, Rs, a=q1, b=q2, c=q3, units=galactic, origin=None, R=rot_mat)

    w0 = gd.PhaseSpacePosition(pos=[pos_init_x, pos_init_y, pos_init_z]*u.kpc,
                               vel=[vel_init_x, vel_init_y, vel_init_z]*u.km/u.s)
    
    n_step = 100
    orbit  = pot_NFW.integrate_orbit(w0, 
                                     dt=t_end*u.Gyr/n_step, 
                                     n_steps=n_step)

    x, y, _ = orbit.xyz.value # kpc

    if type == 'data':
        x_fixed, y_fixed = get_fixed_theta(x, y, n_step)
    elif type == 'model':
        x_fixed, y_fixed = get_fixed_theta(x, y, 10*n_step)

    return x_fixed, y_fixed

class MyWishart(scipy.stats._multivariate.wishart_gen):
    def __init__(self, a,b,c,d,e,f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization if necessary
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _standard_rvs(self, n, shape, dim, df, random_state):
        '''
        Adapted from scipy
        '''
        # Random normal variates for off-diagonal elements from U(0,1)
        n_tril = dim * (dim-1) // 2
        covariances = np.array([self.a,
                                self.b,
                                self.c]).reshape(shape+(n_tril,))
        
        # Random chi-square variates for diagonal elements
        variances = np.array([self.d,
                              self.e,
                              self.f]).reshape((dim,) + shape[::-1]).T
                              
        
        # Create the A matri(ces) - lower triangular
        A = np.zeros(shape + (dim, dim))

        # Input the covariances
        size_idx = tuple([slice(None, None, None)]*len(shape))
        tril_idx = np.tril_indices(dim, k=-1)
        A[size_idx + tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(dim)
        A[size_idx + diag_idx] = variances

        return A
    
def GaussianMixtureModel(x, y, sigma):
    means = np.concatenate( (x[:, None], y[:, None]), axis=1)

    dim = 2
    covariances = np.zeros((len(means), dim, dim)) + np.identity(dim)
    covariances[:,0,0] = sigma**2
    covariances[:,1,1] = sigma**2

    weights = np.zeros(len(means)) + 1/len(means) 
    weights /= weights.sum()

    gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

    return gmm

def get_fixed_theta(x, y, NN):
    r = np.sqrt(x**2 + y**2)
    theta = np.unwrap(np.arctan2(y, x))

    dtheta = abs(np.diff(theta))
    rdtheta = np.cumsum( np.insert(r[:-1]*dtheta, 0, 0) )

    # Fit cubic splines for x and y as functions of theta
    f_x = CubicSpline(rdtheta, x)
    f_y = CubicSpline(rdtheta, y)

    # Generate gamma values, which are evenly spaced theta values for interpolation
    gamma = np.linspace(rdtheta.min(), rdtheta.max(), NN)

    # Evaluate the cubic splines at the gamma values
    theta_x_data = f_x(gamma)
    theta_y_data = f_y(gamma)

    return theta_x_data, theta_y_data
    
def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return Rotation.from_rotvec(angle * v3).as_matrix()

if __name__ == "__main__":
    # Generate Data
    
    ndim  = 15  # Number of dimensions (parameters)
    n_eff = 10000
    seed  = 7345 #np.random.randint(0, 10000)
    np.random.seed(seed)
    sigma = 3
    nlive = 1200

    ### DATA ###
    p0 = np.random.uniform(0, 1, size=ndim)
    theo_params = prior_transform(p0)
    x_data, y_data = model(theo_params, type='data')

    sigma = 3
    x_noise = np.random.normal(0, sigma, len(x_data))
    y_noise = np.random.normal(0, sigma, len(y_data))
    dict_data = {'x': x_data + x_noise,
                 'y': y_data + y_noise,
                 'sigma': sigma}
    
    for i in range(10):
        p      = np.random.uniform(0, 1, size=ndim)
        params = prior_transform(p)
        logl = log_likelihood_GMM(params, dict_data)

        x_model, y_model = model(params, type='model')

        plt.title(f'logL = {logl}')
        plt.scatter(x_model, y_model, s=25, c='k', label='Model')
        plt.scatter(dict_data['x'], dict_data['y'], c='r', label='Data')
        plt.scatter(0,0,c='green', s=10)
        plt.legend(loc='upper right')
        plt.show()

    # # Run and Save Dynesty
    # nworkers = os.cpu_count()
    # pool = Pool(nworkers)

    # sampler = dynesty.DynamicNestedSampler(log_likelihood,
    #                                        prior_transform, 
    #                                        sample='rslice',
    #                                        ndim=ndim, 
    #                                        nlive=nlive,
    #                                        bound='multi',
    #                                        pool=pool, queue_size=nworkers, 
    #                                        logl_args=[dict_data])
    
    # sampler.run_nested(n_effective=n_eff)
    # pool.close()
    # pool.join()
    # results = sampler.results

    # # save_directory = f'./dynesty_results_Sergey_GMM_seed{seed}_sigma{sigma}_ndim{ndim}_nlive{nlive}' 
    # save_directory = f'./dynesty_results_Sergey_seed{seed}_sigma{sigma}_ndim{ndim}_nlive{nlive}' 

    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)

    # # Save parameters
    # params_file = os.path.join(save_directory, 'params.txt')
    # np.savetxt(params_file, theo_params)

    # # Save Dynesty results to a pickle file within the directory
    # results_file = os.path.join(save_directory, 'dynesty_results.pkl')
    # with open(results_file, 'wb') as file:
    #     pickle.dump(results, file)

    # # Save the dictionary of data to a pickle file within the directory
    # data_file = os.path.join(save_directory, 'data_dict.pkl')
    # with open(data_file, 'wb') as file:
    #     pickle.dump(dict_data, file)
