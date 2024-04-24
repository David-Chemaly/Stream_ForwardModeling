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
from scipy.interpolate import interp1d
from scipy.stats import norm, truncnorm
from sklearn.mixture import GaussianMixture

from orbit_evolution_potential import run, run_Gala

### Define Data ###
def generate_data(data_type, sigma, ndim, seed=None):

    np.random.seed(seed)
    utheta = np.random.rand(ndim)
    params = prior_transform(utheta)

    ### Run ###
    clean_data = model(params)
    
    if data_type == 'xy':
        ### Add Noise ### 
        dirty_data = clean_data + np.random.normal(0, sigma, clean_data.shape)

    elif data_type == 'radial':
        ### Add Noise only to Radius ### CHEATING ###
        r   = np.sqrt(clean_data[0]**2 + clean_data[1]**2) + np.random.normal(0, sigma, len(clean_data[0]))
        phi = np.arctan2(clean_data[1], clean_data[0]) 
        dirty_data = np.array([r*np.cos(phi), r*np.sin(phi)])

    return clean_data, dirty_data, params

### Define Functions for Dynesty ###
# Priors
def prior_transform(utheta):
    # Unpack the unit cube values
    u_halo_mass, u_Rs, u_flattening_xy, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, u_kx, u_ky, u_kz = utheta

    logM_min, logM_max     = 11, 12
    logRs_min, logRs_max   = 0.5, 1.5
    q_min, q_max = 0.6, 1
    

    x_pos_min, x_pos_max = -75, -25
    y_pos_min, y_pos_max = -5, 5
    z_pos_min, z_pos_max = -75, 75

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    

    # Transform each parameter
    logM  = logM_min + u_halo_mass * (logM_max - logM_min) 
    logRs = logRs_min + u_Rs * (logRs_max - logRs_min)  
    q     = q_min + u_flattening_xy * (q_max-q_min)  
    
    pos_init_x = x_pos_min + u_pos_init_x * (x_pos_max - x_pos_min) 
    pos_init_y = y_pos_min + u_pos_init_y * (y_pos_max - y_pos_min) 
    pos_init_z = z_pos_min + u_pos_init_z * (z_pos_max - z_pos_min) 

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = abs( norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel) )
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    kx = norm.ppf(u_kx, loc=0, scale=1)
    ky = norm.ppf(u_ky, loc=0, scale=1)
    kz = abs( norm.ppf(u_kz, loc=0, scale=1) )

    # Return the transformed parameters
    return (logM, logRs, q,
            pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z,
            t_end, kx, ky, kz)

# Model
def model(params):
    # Unpack parameters
    logM, logRs, q, \
    pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z, \
    t_end, kx, ky, kz = params

    # Repack some of the parameters to match the expected input format of your 'run' function
    pos_init = np.array([pos_init_x, pos_init_y, pos_init_z]) * u.kpc
    vel_init = np.array([vel_init_x, vel_init_y, vel_init_z]) * u.km/u.s

    # Call your 'run' function
    orbit_pos_p, orbit_pos_N, leading_arg, trailing_arg = run_Gala(10**logM*u.Msun, 10**logRs*u.kpc, q, t_end*u.Gyr, pos_init, vel_init, kx, ky, kz)
    test_pos = orbit_pos_p.T[:2].value 

    x_pos, y_pos = test_pos[0], test_pos[1]

    gamma_pos = get_gamma(test_pos)

    f_x = interp1d(gamma_pos, x_pos, kind='linear')
    f_y = interp1d(gamma_pos, y_pos, kind='linear')

    gamma_new = np.linspace(gamma_pos.min(), gamma_pos.max(), test_pos.shape[1])

    x_new = f_x(gamma_new)
    y_new = f_y(gamma_new)

    return np.array([x_new, y_new])  


def get_gamma(pos):
    x, y = pos[0], pos[1]

    # Calculate differences
    dx = np.diff(x)
    dy = np.diff(y)

    # Euclidean distances between successive points
    distances = np.sqrt(dx**2 + dy**2)

    # Cumulative distance (gamma)
    gamma = np.cumsum(distances)

    # To include the starting point (distance = 0), you might prepend 0 to gamma
    gamma = np.insert(gamma, 0, 0)

    return gamma

# Likelihood
def log_likelihood_gamma(params, dict_data):

    # Unpack the data
    clean_data  = dict_data['clean_data']
    dirty_data  = dict_data['dirty_data']
    sigma = dict_data['sigma']

    # Generate model predictions for the given parameters
    test_pos = model(params)

    data_gamma = get_gamma(clean_data) ### CHEATING BY USING CLEAN DATA ###
    test_gamma = get_gamma(test_pos)
    f_x = interp1d(test_gamma, test_pos[0], kind='linear')
    f_y = interp1d(test_gamma, test_pos[1], kind='linear')

    # Outlier (something really big)
    outlier = 1e2 #1e8

    test_pred = np.zeros(dirty_data.shape)
    if test_gamma[-1] < data_gamma[-1]:

        len_in    = np.sum(data_gamma < test_gamma[-1])
        test_pred[0,:len_in] = f_x(data_gamma[:len_in])
        test_pred[1,:len_in] = f_y(data_gamma[:len_in])

        # Compute residuals after fit and punishement for being too short
        residuals   = dirty_data[:, :len_in] - test_pred[:,:len_in]
        punishement = outlier * (data_gamma[-1] - test_gamma[-1])

    elif test_gamma[-1] > data_gamma[-1]:

        test_pred[0] = f_x(data_gamma)
        test_pred[1] = f_y(data_gamma)

        # Compute residuals after fit and punishement for going beyond data
        residuals   = dirty_data - test_pred
        punishement = outlier * (test_gamma[-1] - data_gamma[-1])

    else:
        # Compute residuals after fit
        residuals   = dirty_data - test_pos
        punishement = 0

    # Assuming 'sigma' is either a scalar or an array of standard deviations for your observed data
    # Chi-squared statistic
    chi_squared = np.sum((residuals / sigma) ** 2)

    # Convert chi-squared to log-likelihood
    # Note: This assumes Gaussian errors. Modify if your error distribution is different.
    log_likelihood = -0.5 * chi_squared - punishement
    
    return log_likelihood

def log_likelihood_density(params, dict_data):

    # Unpack the data
    clean_data  = dict_data['clean_data']
    dirty_data  = dict_data['dirty_data']
    sigma = dict_data['sigma']

    # Generate model predictions for the given parameters
    model_data = model(params)

    # Based on FOV of data
    x_min, x_max = -120, -20
    y_min, y_max = 10, 100

    # Based on spatial resolution
    reso = 30

    im_data  = np.histogram2d(dirty_data[0], dirty_data[1], bins=reso, range=[[x_min, x_max], [y_min, y_max]])[0]
    im_model = np.histogram2d(model_data[0], model_data[1], bins=reso, range=[[x_min, x_max], [y_min, y_max]])[0]

    if im_model.sum() != im_data.sum():
        log_likelihood = -np.inf

    else:
        # Assuming 'sigma' is either a scalar or an array of standard deviations for your observed data
        # Chi-squared statistic
        chi_squared = np.sum(((im_data - im_model) / sigma) ** 2)

        # Convert chi-squared to log-likelihood
        # Note: This assumes Gaussian errors. Modify if your error distribution is different.
        log_likelihood = -0.5 * chi_squared
    
    return log_likelihood

def unwrap(angles):
    arg_decrease = np.where( np.diff(angles) <= 0 )[0]
    for i in arg_decrease:
        angles[i+1:] += 2 * np.pi

    return angles

def log_likelihood_phi(params, dict_data):

    # Unpack the data
    clean_data  = dict_data['clean_data']
    dirty_data  = dict_data['dirty_data']
    sigma = dict_data['sigma']

    # Generate model predictions for the given parameters
    model_data = model(params)

    r_data   = np.sqrt(dirty_data[0]**2 + dirty_data[1]**2)
    phi_data = unwrap(np.arctan2(dirty_data[1], dirty_data[0]))

    r_model   = np.sqrt(model_data[0]**2 + model_data[1]**2)
    phi_model = unwrap(np.arctan2(model_data[1], model_data[0]))

    if phi_data.min() != phi_model.min() or phi_data.max() != phi_model.max() or len(phi_model) != len(phi_data):
        log_likelihood = -np.inf

    else:
        f_r   = interp1d(phi_model, r_model, kind='linear')
        r_new = f_r(phi_data)

        # Assuming 'sigma' is either a scalar or an array of standard deviations for your observed data
        # Chi-squared statistic
        chi_squared = np.sum(((r_data - r_new) / sigma) ** 2)

        # Convert chi-squared to log-likelihood
        # Note: This assumes Gaussian errors. Modify if your error distribution is different.
        log_likelihood = -0.5 * chi_squared
    
    return log_likelihood

def log_likelihood_GMM(params, dict_data):

    # Unpack the data
    clean_data  = dict_data['clean_data']
    dirty_data  = dict_data['dirty_data']
    sigma = dict_data['sigma']

    # Generate model predictions for the given parameters
    model_data = model(params)

    # Probability of CDF
    x_model_data = model_data[0]
    y_model_data = model_data[1]

    # Define means and covariances for GMM
    means = np.concatenate( (x_model_data[:, None], y_model_data[:, None]), axis=1)

    dim = 2
    covariances = np.zeros((len(means), dim,dim)) + np.identity(dim)
    covariances[:,0,0] = sigma**2
    covariances[:,1,1] = sigma**2

    # Define mixing coefficients (weights)
    weights = np.zeros(len(means)) + 1/len(means)  # Replace with actual values

    # Ensure that weights sum to 1
    weights /= weights.sum()

    # Create Gaussian Mixture Model
    gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

    # Points to evaluate (as an array of [x, y])
    points = np.concatenate( (dirty_data[0, :, None], dirty_data[1, :, None]), axis=1)

    # Compute the probability density at these points
    probabilities = np.exp(gmm.score_samples(points))

    # Log likelihood
    if np.where(probabilities <= 1e-300)[0].size != 0:
        log_likelihood = -1e11
    else:
        log_likelihood = np.sum(np.log(probabilities))

    return log_likelihood

def log_likelihood_fixed(params, dict_data):
    return -100.

if __name__ == "__main__":
    # Generate Data
    
    ndim  = 13  # Number of dimensions (parameters)
    seed  = 340
    sigma = 3
    nlive = 1000
    clean_data, dirty_data, theo_params = generate_data(data_type='xy', sigma=sigma, ndim=ndim, seed=seed)
    dict_data = {'clean_data': clean_data, 'dirty_data': dirty_data, 'sigma': sigma}

    # Run Dynesty
    nworkers = os.cpu_count()
    pool = Pool(nworkers)

    sampler = dynesty.DynamicNestedSampler(log_likelihood_GMM,
                                           prior_transform, 
                                           sample='rslice',
                                           ndim=ndim, 
                                           nlive=nlive,
                                           bound='multi',
                                           pool=pool, queue_size=nworkers, 
                                           logl_args=[dict_data])
    
    sampler.run_nested()
    pool.close()
    pool.join()
    results = sampler.results

    save_directory = f'./dynesty_results_GMM_seed{seed}_sigma{sigma}_ndim{ndim}_nlive{nlive}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save parameters
    params_file = os.path.join(save_directory, 'params.txt')
    np.savetxt(params_file, theo_params)

    # Save Dynesty results to a pickle file within the directory
    results_file = os.path.join(save_directory, 'dynesty_results.pkl')
    with open(results_file, 'wb') as file:
        pickle.dump(results, file)

    # Save the dictionary of data to a pickle file within the directory
    data_file = os.path.join(save_directory, 'data_dict.pkl')
    with open(data_file, 'wb') as file:
        pickle.dump(dict_data, file)