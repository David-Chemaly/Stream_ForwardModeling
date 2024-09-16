import os
import h5py
import pickle
import dynesty
import numpy as np
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

import scipy
from scipy.stats import norm, chi2
from scipy.integrate import quad
from sklearn.mixture import GaussianMixture

### Prior transform function ###
def prior_transform(utheta):
    # Unpack the unit cube values
    u_logM, u_Rs, \
    u_logm, u_rs, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, u_a, u_b, u_c, u_kx, u_ky, u_kz = utheta

    logM_min, logM_max     = 11, 12
    Rs_min, Rs_max         = 3, 30
    
    logm_min, logm_max     = 7, 8
    rs_min, rs_max         = 1, 10

    mean_pos = 0
    std_pos = 100

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  
    
    logm  = logm_min + u_logm * (logm_max - logm_min) 
    rs    = rs_min + u_rs * (rs_max - rs_min)  

    pos_init_x = norm.ppf(u_pos_init_x, loc=mean_pos, scale=std_pos)
    pos_init_y = norm.ppf(u_pos_init_y, loc=mean_pos, scale=std_pos)
    pos_init_z = norm.ppf(u_pos_init_z, loc=mean_pos, scale=std_pos)

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel) 
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
    return (logM, Rs, logm, rs,
            pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z,
            t_end, a, b, c, kx, ky, kz)

def log_likelihood_gaussians(params, dict_data):

    # Unpack the data
    r_data     = dict_data['r']
    theta_data = dict_data['theta']
    r_sig      = dict_data['r_sigma']
    theta_sig  = dict_data['theta_sigma']

    # Generate model predictions for the given parameters
    x_model, y_model = model(params)

    r_model     = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.arctan2(y_model, x_model)

    likelihood = 0
    for i in range(len(r_data)):
        theta_min = theta_data[i] - theta_sig
        theta_max = theta_data[i] + theta_sig

        arg_in = np.where((theta_model > theta_min) & (theta_model < theta_max))[0]
        if len(arg_in) <= 1:
            return -1e100
        
        r_in   = r_model[arg_in]

        r_mean_stream = np.mean(r_in)
        r_sig_stream  = np.std(r_in)

        likelihood += np.log( overlap_area(r_data[i], r_sig, r_mean_stream, r_sig_stream) )

    return likelihood

def log_likelihood_GMM(params, dict_data):

    # Unpack the data
    x_data = dict_data['x']
    y_data = dict_data['y']
    sigma  = dict_data['sigma']

    # Generate model predictions for the given parameters
    x_model, y_model = model(params)

    gmm = GaussianMixtureModel(x_model, y_model, sigma)

    samples = np.concatenate((x_data[:, None], y_data[:, None]), axis=1)
    logl    = np.sum( gmm.score_samples(samples) )

    return logl

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

def spheroid_params(a, b, c, kx, ky, kz):
    # Wishart Prior
    df    = 3
    scale = np.identity(df)

    my_wishart        = MyWishart(a,b,c,kx,ky,kz)
    covariance_matrix = my_wishart.rvs(df, scale)

    eigvals, eigvec   = np.linalg.eigh(covariance_matrix)

    q1, q2, q3 = eigvals**0.5
    rot_mat    = eigvec

    return q1, q2, q3, rot_mat

def model(params):
    # Unpack parameters
    logM, Rs, logm, rs, \
    x0, y0, z0, vx0, vy0, vz0, \
    t_end, a, b, c, kx, ky, kz = params

    q1, q2, q3, rot_mat = spheroid_params(a, b, c, kx, ky, kz)
    
    # Run the stream generator
    stream_x, stream_y = stream_gen_Gala(logM, Rs*u.kpc, q1, q2, q3, logm, rs*u.kpc, x0, y0, z0, vx0, vy0, vz0, t_end, rot_mat)

    return stream_x, stream_y

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
    
def stream_gen_Gala(logM, Rs,
                    q1, q2, q3,
                    logm, rs,
                    x0, y0, z0,
                    vx0, vy0, vz0,
                    time, 
                    rot_matrix,
                    dt = 1*u.Myr):
    
    pot = gp.NFWPotential(10**logM*u.Msun, 
                          Rs, 
                          a=1, b=q2/q3, c=q1/q3, 
                          units=galactic, 
                          origin=None, 
                          R=None)

    H = gp.Hamiltonian(pot)

    prog_w0 = gd.PhaseSpacePosition(pos=np.array([x0, y0, z0]) * u.kpc,
                                    vel=np.array([vx0, vy0, vz0]) * u.km/u.s)

    df = ms.FardalStreamDF(gala_modified=True, lead=True, trail=True)

    prog_pot = gp.PlummerPotential(m=10**logm*u.Msun, 
                                   b=rs, 
                                   units=galactic)

    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    orbit = pot.integrate_orbit(prog_w0, dt=dt, n_steps=(time*u.Gyr)//dt)
    stream, _ = gen.run(prog_w0, 
                        10**logm*u.Msun,
                        dt=dt, 
                        n_steps=(time*u.Gyr)//dt)
    
    x_stream, y_stream, _ = rot_matrix @ stream.xyz.value
    x_orbti, y_orbit, _ = rot_matrix @ orbit.xyz.value

    return x_orbti, y_orbit, x_stream, y_stream

# Function to compute the overlapping area of two Gaussian distributions
def overlap_area(mu1, sigma1, mu2, sigma2):
    def gauss1(x):
        return norm.pdf(x, mu1, sigma1)
    def gauss2(x):
        return norm.pdf(x, mu2, sigma2)
    def min_gauss(x):
        return min(gauss1(x), gauss2(x))
    
    # Integration limits based on the means and standard deviations
    lower_limit = min(mu1 - 4*sigma1, mu2 - 4*sigma2)
    upper_limit = max(mu1 + 4*sigma1, mu2 + 4*sigma2)
    
    # Numerical integration over the minimum of both Gaussian PDFs
    result, _ = quad(min_gauss, lower_limit, upper_limit)
    return result

if __name__ == "__main__":
    # Generate Data
    
    ndim  = 17  # Number of dimensions (parameters)
    n_eff = 10000
    seed  = 8940
    sigma = 3
    nlive = 1100

    ### DATA ###
    size_data = 10

    r_data     = 50*np.ones(size_data)
    r_sig      = 1
    theta_data = np.linspace(0, np.pi, size_data)
    theta_sig  = (theta_data[1] - theta_data[0]) / 2

    dict_data = {'r': r_data, 
                 'theta': theta_data, 
                 'r_sigma': r_sig,
                 'theta_sigma': theta_sig}
    

    # Just testing
    for i in range(50):
        p      = np.random.uniform(0, 1, size=ndim)
        params = prior_transform(p)
        x_stream, y_stream = model(params)
        logl = log_likelihood_gaussians(params, dict_data)

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.xlabel('x [kpc]', fontsize=15)
        plt.ylabel('y [kpc]', fontsize=15)
        plt.scatter(x_stream, y_stream, s=1, c='k')
        x_data = r_data * np.cos(theta_data)
        y_data = r_data * np.sin(theta_data)
        plt.scatter(x_data, y_data, s=10, c='r')
        plt.subplot(1,2,2)
        plt.xlabel(r'$\theta$ [rad]', fontsize=15)
        plt.ylabel(r'$r$ [kpc]', fontsize=15)
        r_stream = np.sqrt(x_stream**2 + y_stream**2)
        theta_stream = np.arctan2(y_stream, x_stream)
        plt.scatter(theta_stream, r_stream, s=1, c='k', label='Model')
        plt.scatter(theta_data, r_data, s=10, c='r', label='Data')
        plt.title(f'logL = {logl:.3e}', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()

    # # Run and Save Dynesty
    # nworkers = os.cpu_count()
    # pool = Pool(nworkers)

    # sampler = dynesty.DynamicNestedSampler(log_likelihood_gaussians,
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

    # save_directory = f'./dynesty_results_GMM_seed{seed}_sigma{sigma}_ndim{ndim}_nlive{nlive}'
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