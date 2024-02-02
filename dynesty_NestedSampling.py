import os
import h5py
import pickle
import dynesty
import numpy as np
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.stats import norm, truncnorm
from sklearn.mixture import GaussianMixture

from orbit_evolution_potential import run

### Define Data ###
def generate_data(data_type, ndim, seed=None):

    if seed is None:
        # 4 for Potential
        halo_mass     = 7.5e11# * u.Msun 
        concentration = 20
        flattening_xy  = 0.75
        flattening_xz  = 1.25

        # 6 for Initial Conditions
        pos_init = [50, -30, 40]# * u.kpc 
        vel_init = [90,100,80]# * u.km/u.s

        # 1 for time
        t_end = 1.5

        # 3 for orientation
        alpha, beta, charlie = 0.01, -0.5, 0.7

        # 2 for rotation
        aa, bb = -0.1, 0.3
    
        params = (halo_mass, concentration, flattening_xy, flattening_xz, pos_init[0], pos_init[1], pos_init[2], vel_init[0], vel_init[1], vel_init[2], t_end, alpha, beta, charlie, aa, bb)
    
    else:
        np.random.seed(seed)
        utheta = np.random.rand(ndim)
        params = prior_transform(utheta)

    ### Run ###
    clean_data = model(params)
    
    if data_type == 'xy':
        ### Add Noise ### 
        sigma = 2
        dirty_data = clean_data + np.random.normal(0, sigma, clean_data.shape)
    elif data_type == 'radial':
        ### Add Noise only to Radius ### CHEATING ###
        sigma = 2
        r   = np.sqrt(clean_data[0]**2 + clean_data[1]**2) + np.random.normal(0, sigma, len(clean_data[0]))
        phi = np.arctan2(clean_data[1], clean_data[0]) 
        dirty_data = np.array([r*np.cos(phi), r*np.sin(phi)])

    return clean_data, dirty_data, sigma, params

### Define Functions for Dynesty ###
# Priors
def prior_transform(utheta):
    # Unpack the unit cube values
    u_halo_mass, u_concentration, u_flattening_xy, u_flattening_xz, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, u_alpha, u_beta, u_charlie, u_aa, u_bb  = utheta

    M_min, M_max     = 1e11, 1e12
    c_min, c_max     = 18, 22
    qxy_min, qxy_max = 0.5, 1.5
    qxz_min, qxz_max = 0.5, 1.5

    x_pos_min, x_pos_max = 25, 75
    y_pos_min, y_pos_max = -75, -25
    z_pos_min, z_pos_max = 25, 75

    x_vel_min, x_vel_max = 75, 125
    y_vel_min, y_vel_max = 75, 125
    z_vel_min, z_vel_max = 75, 125

    t_end_min, t_end_max = 1, 2

    alpha_mu, alpha_sigma = 0, 1
    beta_mu, beta_sigma = 0, 1
    charlie_mu, charlie_sigma = 0, 1
    aa_mu, aa_sigma = 0, 1
    bb_mu, bb_sigma = 0, 1

    # Transform each parameter
    halo_mass     = M_min + u_halo_mass * (M_max - M_min) 
    concentration = c_min + u_concentration * (c_max - c_min)  
    flattening_xy = qxy_min + u_flattening_xy * (qxy_max-qxy_min)  
    flattening_xz = qxz_min + u_flattening_xz * (qxz_max-qxz_min)  

    pos_init_x = x_pos_min + u_pos_init_x * (x_pos_max - x_pos_min) 
    pos_init_y = y_pos_min + u_pos_init_y * (y_pos_max - y_pos_min) 
    pos_init_z = z_pos_min + u_pos_init_z * (z_pos_max - z_pos_min) 

    vel_init_x = x_vel_min + u_vel_init_x * (x_vel_max - x_vel_min) 
    vel_init_y = y_vel_min + u_vel_init_y * (y_vel_max - y_vel_min) 
    vel_init_z = z_vel_min + u_vel_init_z * (z_vel_max - z_vel_min) 

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    alpha = norm.ppf(u_alpha, loc=alpha_mu, scale=alpha_sigma)
    beta = norm.ppf(u_beta, loc=beta_mu, scale=beta_sigma)
    charlie = norm.ppf(u_charlie, loc=charlie_mu, scale=charlie_sigma)
    aa = norm.ppf(u_aa, loc=aa_mu, scale=aa_sigma)
    bb = norm.ppf(u_bb, loc=bb_mu, scale=bb_sigma)

    # Return the transformed parameters
    return (halo_mass, concentration, flattening_xy, flattening_xz,
            pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z,
            t_end, alpha, beta, charlie, aa, bb)

# Model
def model(params):
    # Unpack parameters
    halo_mass, concentration, flattening_xy, flattening_xz, \
    pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z, \
    t_end, alpha, beta, charlie, aa, bb = params

    # Repack some of the parameters to match the expected input format of your 'run' function
    pos_init = np.array([pos_init_x, pos_init_y, pos_init_z]) * u.kpc
    vel_init = np.array([vel_init_x, vel_init_y, vel_init_z]) * u.km/u.s

    # Call your 'run' function
    N_time   = 100
    test_pos = run(halo_mass*u.Msun, concentration, flattening_xy, flattening_xz, pos_init, vel_init, t_end, alpha, beta, charlie, aa, bb, N_time)[:2].value 

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
    covariances[:,0,0] = sigma
    covariances[:,1,1] = sigma

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
        log_likelihood = -np.inf
    else:
        log_likelihood = np.sum(np.log(probabilities))

    return log_likelihood

if __name__ == "__main__":
    # Generate Data
    
    ndim = 16  # Number of dimensions (parameters)
    seed = 99
    clean_data, dirty_data, sigma, theo_params = generate_data(data_type='xy', ndim=ndim, seed=seed)
    dict_data = {'clean_data': clean_data, 'dirty_data': dirty_data, 'sigma': sigma}

    # Run Dynesty
    nworkers = os.cpu_count()
    pool = Pool(nworkers)

    sampler = dynesty.DynamicNestedSampler(log_likelihood_GMM, prior_transform, ndim, pool=pool, queue_size=nworkers, logl_args=[dict_data])
    sampler.run_nested()
    pool.close()
    pool.join()
    results = sampler.results

    save_directory = f'./dynesty_results_N100_GMM_seed{seed}'
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