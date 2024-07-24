import os
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import h5py
import time
import pickle
import dynesty
import argparse
import contextlib
import numpy as np
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dynesty import plotting as dyplot

import scipy
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm, linregress
from scipy.spatial.transform import Rotation as R
from sklearn.mixture import GaussianMixture

import astropy.units as u
from astropy.constants import G
G = G.to(u.pc * u.Msun**-1 * (u.km / u.s)**2)

import agama
# working units: 1 Msun, 1 kpc, 1 km/s
agama.setUnits(length=1, velocity=1, mass=1)
from Agama_dynesty import model as stream_model
from Agama_dynesty import prior_transform as stream_prior_transform

BAD_VAL = 1e50
SUPER_BAD_VAL = 1e80

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * np.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm

### Unwrap function ###
def unwrap(r, theta, gamma):
    theta[theta < 0] += 2*np.pi
    unwrapped_theta = np.unwrap(theta)
    if np.any(unwrapped_theta < 0):
        return np.flip(r), np.unwrap(np.flip(theta)), np.flip(gamma)
    else:
        return r, unwrapped_theta, gamma
    
### Prior transform function ###
def prior_transform(utheta):
    # Unpack the unit cube values
    u_logM, u_Rs, u_p, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, \
    u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, \
    u_x1, u_x2, u_x3 = utheta

    logM_min, logM_max = 11, 13
    Rs_min, Rs_max     = 5, 25

    mean_pos = 0
    std_pos  = 100

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  
    p = u_p   

    pos_init_x = abs(norm.ppf(u_pos_init_x, loc=mean_pos, scale=std_pos))
    pos_init_y = abs(norm.ppf(u_pos_init_y, loc=mean_pos, scale=std_pos/100))
    pos_init_z = norm.ppf(u_pos_init_z, loc=mean_pos, scale=std_pos)

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = abs(norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel))
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    x1 = u_x1
    x2 = u_x2
    x3 = u_x3

    # Return the transformed parameters
    return (logM, Rs, p,
            pos_init_x, pos_init_y, pos_init_z, 
            vel_init_x, vel_init_y, vel_init_z,
            t_end, 
            x1, x2, x3)

def get_rot_mat(x1, x2, x3):
    V = np.array([np.cos(2*np.pi*x2)*np.sqrt(x3), 
                  np.sin(2*np.pi*x2)*np.sqrt(x3), 
                  np.sqrt(1-x3)])
    
    I = np.identity(3)

    H = I - 2 * V[:, None] @ V[None]

    R = np.array([[ np.cos(2*np.pi*x1), np.sin(2*np.pi*x1), 0],
                  [-np.sin(2*np.pi*x1), np.cos(2*np.pi*x1), 0],
                  [0,0,1]])

    M = -H @ R

    return M

def log_likelihood_MSE(params, dict_data):

    # Generate model predictions for the given parameters
    xy_model    = model(params)
    r_model     = np.sqrt(xy_model[0]**2 + xy_model[1]**2)
    theta_model = np.arctan2(xy_model[1], xy_model[0])
    r_model, theta_model, _ = unwrap(r_model, theta_model, None)

    sign = (np.diff(theta_model) <= 0).sum() 
    if sign != 0:
        logl = -SUPER_BAD_VAL * sign
    else: 
        f = interp1d(theta_model, r_model, kind='cubic', bounds_error=False, fill_value=np.nan)
        r_fit = f(dict_data['theta'])

        # Punish for too short
        N_nan = np.sum(np.isnan(r_fit))

        if N_nan > 0:
            logl = -BAD_VAL *  N_nan
        else:
            logl = -0.5*np.sum( (dict_data['r'] - r_fit)**2/dict_data['r_sigma']**2 + np.log(dict_data['r_sigma']**2) )

    return logl

def model(params, dt=0.001):
    # Unpack parameters
    logM, Rs, p, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end, \
    x1, x2, x3 = params

    rot_mat = get_rot_mat(x1, x2, x3)
    rot = R.from_matrix(rot_mat)
    euler_angles = rot.as_euler('xyz', degrees=False)

    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, p, 1)

    # Set host potential
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, 
                               axisRatioY=p, axisRatioZ=1, orientation=euler_angles)

    # initial displacement
    r_center = np.array([pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z])

    _, xv = agama.orbit(ic=r_center, potential=pot_host, time=t_end, timestart=0, trajsize=int(t_end/dt)+1, verbose=False)

    xy_orbit = xv[:, :2].T

    return xy_orbit

    
def get_data_track(ndim, model, prior_transform, N_data=20, seed=False, mean_p=None, mean_q=None, std_p=None, std_q=None, min_length=10, max_length=100, min_theta=np.pi/2, max_theta=5*np.pi/2):

    d_length = 0
    theta_length = 0
    theta_track_data = np.array([0])
    while d_length < min_length or theta_length < min_theta or max_length < d_length or max_theta < theta_length or np.any( np.diff(theta_track_data) < 0): 

        if seed:
            params_data = np.array([  11.68673801,    5.33486954,    0.9       ,    0.7       ,
                                    8.13813955,    3.7778778 ,  -12.23542721,    0.90062788,
                                    -68.02825117,  -73.27557004, -101.99732061,   -3.17882779,
                                    2.22963491,    0.29847474,    0.59976523,    0.15987843])
            
        else:
            p = np.random.uniform(0, 1, ndim)
            params_data = np.array(prior_transform(p))
            if mean_p!=None or mean_q!=None:
                if std_p==None or std_q==None:
                    params_data[2] = mean_p
                    params_data[3] = mean_q
                else: 
                    params_data[2] = np.random.normal(mean_p, std_p)
                    params_data[3] = np.random.normal(mean_q, std_q)


        xy_stream_data, xy_track_data, gamma, gamma_track = model(params_data, N_stars=500, Nbody=False, seed=False)
        r_stream_data = np.sqrt(xy_stream_data[0]**2 + xy_stream_data[1]**2)
        r_track_data = np.sqrt(xy_track_data[0]**2 + xy_track_data[1]**2)
        theta_track_data = np.arctan2(xy_track_data[1], xy_track_data[0])
        r_track_data, theta_track_data, gamma_track = unwrap(r_track_data, theta_track_data, gamma_track)

        arc_lengths = abs(r_track_data[:-1] * np.diff(theta_track_data))
        cumulative_arc_lengths = np.concatenate([[0], np.cumsum(arc_lengths)])
        d_length     = cumulative_arc_lengths[-1]
        theta_length = abs(theta_track_data.max() - theta_track_data.min())

    # Get full N-body data
    xy_stream_data, xy_track_data, gamma, gamma_track = model(params_data, N_stars=10000, Nbody=True, seed=False)
    r_stream_data = np.sqrt(xy_stream_data[0]**2 + xy_stream_data[1]**2)
    r_track_data = np.sqrt(xy_track_data[0]**2 + xy_track_data[1]**2)
    theta_track_data = np.arctan2(xy_track_data[1], xy_track_data[0])
    r_track_data, theta_track_data, gamma_track = unwrap(r_track_data, theta_track_data, gamma_track)

    d = d_length/N_data  # kpc -- Desired fixed distance
    fixed_distances = np.arange(d, cumulative_arc_lengths[-1]-d, d)

    interp_theta = interp1d(cumulative_arc_lengths, theta_track_data, kind='cubic')
    interp_gamma = interp1d(cumulative_arc_lengths, gamma_track, kind='cubic')
    theta_data = interp_theta(fixed_distances)
    gamma_data = interp_gamma(fixed_distances)

    f = interp1d(theta_track_data, r_track_data, kind='cubic')
    r_data = f(theta_data)

    N_data  = []
    N_pred  = []
    r_sigma = []
    for i in range(len(gamma_data)):
        if i == 0:
            dgamma = (gamma_data[i+1] - gamma_data[i])/2
            gamma_1, gamma_2 = gamma_data[i]-dgamma, gamma_data[i]+dgamma
            gamma_min, gamma_max = min(gamma_1, gamma_2), max(gamma_1, gamma_2)
        elif i == len(gamma_data)-1:
            dgamma = (gamma_data[i] - gamma_data[i-1])/2
            gamma_1, gamma_2 = gamma_data[i]-dgamma, gamma_data[i]+dgamma
            gamma_min, gamma_max = min(gamma_1, gamma_2), max(gamma_1, gamma_2)
        else:
            dgamma_min = (gamma_data[i] - gamma_data[i-1])/2
            dgamma_max = (gamma_data[i+1] - gamma_data[i])/2
            gamma_1, gamma_2 = gamma_data[i]-dgamma_min, gamma_data[i]+dgamma_max
            gamma_min, gamma_max = min(gamma_1, gamma_2), max(gamma_1, gamma_2)
        arg_in = np.where((gamma>gamma_min) & (gamma<gamma_max))[0]
        N_data.append(len(arg_in))
        N_pred.append(len(arg_in) * 500/10000)
        r_sigma.append(np.std(r_stream_data[arg_in]))
    N_data = np.array(N_data)
    N_pred = np.array(N_pred)
    r_sigma = np.array(r_sigma)#/np.sqrt(N_data)

    x_data = r_data * np.cos(theta_data)
    y_data = r_data * np.sin(theta_data)

    dict_data = {'theta': theta_data,
                'r': r_data,
                'r_sigma': r_sigma,
                'x': x_data,
                'y': y_data}
    
    return dict_data, params_data

def get_data_orbit(ndim, model, sigma=1, N_data=20):
    not_yet = True
    while not_yet:
        p = np.random.uniform(0, 1, ndim)
        params = prior_transform(p)
        x_data, y_data = model(params)

        r_data     = np.sqrt(x_data**2 + y_data**2)
        theta_data = np.arctan2(y_data, x_data)
        r_data, theta_data, _ = unwrap(r_data, theta_data, None)

        arc_lengths = abs(r_data[:-1] * np.diff(theta_data))
        cumulative_arc_lengths = np.concatenate([[0], np.cumsum(arc_lengths)])
        d_length     = cumulative_arc_lengths[-1]
        theta_length = abs(theta_data.max() - theta_data.min())

        sign = (np.diff(theta_data) <= 0).sum() 

        _, _, r_value, _, _ = linregress(x_data, y_data)

        if sign == 0 and d_length > 30 and d_length < 100 and theta_length > np.pi/2 and theta_length < 5*np.pi/2 and abs(r_value) < 0.95:
            not_yet = False

    d = d_length/N_data  # kpc -- Desired fixed distance
    fixed_distances = np.arange(d, cumulative_arc_lengths[-1]-d, d)

    interp_gamma = interp1d(cumulative_arc_lengths, theta_data, kind='cubic')
    theta_fit = interp_gamma(fixed_distances)

    interp_theta = interp1d(theta_data, r_data, kind='cubic')
    r_fit = interp_theta(theta_fit)
    noise = np.random.normal(0, sigma, len(r_fit))
    r_fit = r_fit + noise

    x_fit = r_fit * np.cos(theta_fit)
    y_fit = r_fit * np.sin(theta_fit)

    dict_data = {'theta': theta_fit,
                'r': r_fit,
                'r_sigma': np.zeros(len(r_fit))+sigma,
                'x': x_fit,
                'y': y_fit}
    
    return dict_data

if __name__ == "__main__":
    # Path to save
    PATH_SAVE = '/data/dc824-2/orbit_to_orbit_fit_2D/p0.8'

    # Hyperparameters
    ndim   = 13
    nlive  = 1200
    N_data = 20
    sigma  = 3

    # Save params
    save_directory = f'{PATH_SAVE}/Stream_1_NoPrior' 

    # Get Data
    params_data = np.array([ 1.16035104e+01,  1.01057301e+01,  8.00000000e-01,
        9.55518405e+01,  2.03007151e-02, -1.58314475e+02, -1.07160671e+00,
        3.19997489e+01,  3.19198798e+01,  1.70428402e+00,  3.06655489e-01,
        9.21862551e-01,  9.54528746e-01])
    dict_data = get_data_orbit(params_data, model, sigma=sigma)
    # dict_data, params_data = get_data_track(ndim+2, stream_model, stream_prior_transform, N_data, seed=True)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    # Save True parameters
    params_file = os.path.join(save_directory, 'true_params.txt')
    np.savetxt(params_file, params_data)
    # params_data = np.concatenate([params_data[:4], params_data[6:]])

    # Save the dictionary of data to a pickle file within the directory
    data_file = os.path.join(save_directory, 'data_dict.pkl')
    with open(data_file, 'wb') as file:
        pickle.dump(dict_data, file)

    # Run and Save Dynesty
    nworkers = os.cpu_count()
    pool     = Pool(nworkers)
    sampler = dynesty.DynamicNestedSampler(log_likelihood_MSE,
                                           prior_transform, 
                                           sample='rslice',
                                           ndim=ndim, 
                                           nlive=nlive,
                                           bound='multi',
                                           pool=pool, 
                                           queue_size=nworkers, 
                                           logl_args=[dict_data])
    sampler.run_nested()
    pool.close()
    pool.join()
    results = sampler.results

    # Save Dynesty results to a pickle file within the directory
    results_file = os.path.join(save_directory, 'dynesty_results.pkl')
    with open(results_file, 'wb') as file:
        pickle.dump(results, file)

    # Plot a subset of parameters
    labels = [r'logM$_{halo}$', r'R$_s$', 
              'p', 
              r'x$_0$', r'y$_0$', r'z$_0$', 
              r'v$_x$', r'v$_y$', r'v$_z$', 
              'time', 
              r'k$_1$', r'k$_2$', r'k$_3$' ]

    # Plot the posteriors
    fig, axes = dyplot.cornerplot(results, 
                                  color='blue',
                                  truths=params_data, 
                                  truth_color='lime',
                                  labels=labels, 
                                  max_n_ticks=5,
                                  show_titles=True)
    plt.savefig(f'{save_directory}/posteriors.png')

    # Plot the best fit

    # Extract weighted samples
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])

    # Compute the weighted mean of the samples
    mean_fit_params = np.sum(samples * weights.reshape(-1, 1), axis=0) / np.sum(weights)

    # Compute the maximum log-likelihood
    max_logl_index = np.argmax(results.logl)
    max_logl_sample = results.samples[max_logl_index]
    max_logl_value = results.logl[max_logl_index]

    # Save MAP parameters
    np.savetxt(os.path.join(save_directory, 'MAP_params.txt'), max_logl_sample)

    max_fit  = model(max_logl_sample)
    theo_fit = model(params_data)

    x_data, y_data = dict_data['x'], dict_data['y']

    # Plot the best fit
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.title(f'MAP: {log_likelihood_MSE(max_logl_sample, dict_data)}')    
    x_data = dict_data['x']
    y_data = dict_data['y']
    theta_data = dict_data['theta']
    r_sigma = dict_data['r_sigma']
    x_sigma = abs(r_sigma*np.cos(theta_data))
    y_sigma = abs(r_sigma*np.sin(theta_data))
    for i in range(len(theta_data)):
        xerr = r_sigma[i] * np.cos(theta_data[i])
        yerr = r_sigma[i] * np.sin(theta_data[i])
        plt.plot([x_data[i] - xerr, x_data[i] + xerr], [y_data[i] - yerr, y_data[i] + yerr], 'lime')
    plt.scatter(x_data, y_data, c='lime')
    plt.plot(max_fit[0], max_fit[1], 'b')
    plt.scatter(0,0, c='k', s=100)
    plt.subplot(1,2,2)
    plt.title(f'True: {log_likelihood_MSE(params_data, dict_data)}')
    for i in range(len(theta_data)):
        xerr = r_sigma[i] * np.cos(theta_data[i])
        yerr = r_sigma[i] * np.sin(theta_data[i])
        plt.plot([x_data[i] - xerr, x_data[i] + xerr], [y_data[i] - yerr, y_data[i] + yerr], 'lime')
    plt.scatter(x_data, y_data, c='lime')    
    plt.plot(theo_fit[0], theo_fit[1], 'b')
    plt.scatter(0,0, c='k', s=100)

    plt.xlabel(r'x [kpc]')
    plt.ylabel(r'y [kpc]')
    plt.savefig(f'{save_directory}/best_fit.png')

    # Plot flattening
    p_samples = results['samples'][:, 2].T

    plt.figure(figsize=(15,5))
    plt.xlabel('p')
    plt.ylabel('Counts')
    plt.hist(p_samples, bins=10, color='b')
    plt.axvline(np.mean(p_samples), c='k')
    plt.axvline(np.mean(p_samples)-np.std(p_samples), c='k',linestyle='--')
    plt.axvline(np.mean(p_samples)+np.std(p_samples), c='k',linestyle='--')
    plt.axvline(params_data[2], c='lime')

    plt.savefig(f'{save_directory}/flattening_posteriors.png')