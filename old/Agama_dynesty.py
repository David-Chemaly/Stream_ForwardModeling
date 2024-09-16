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
plt.rcParams.update({'font.size': 16})
from multiprocessing import Pool
from dynesty import plotting as dyplot

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

from Agama_stream import Agama_stream

BAD_VAL = 1e50

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
    u_logM, u_Rs, u_p, u_q, \
    u_logm, u_rs, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, \
    u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, \
    u_x1, u_x2, u_x3 = utheta

    logM_min, logM_max = 11, 13
    Rs_min, Rs_max     = 5, 25

    logm_min, logm_max = 8, 10
    rs_min, rs_max     = 1, 5

    mean_pos = 0
    std_pos  = 100

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  
    p = u_p
    q = u_q
    # p     = 1 - np.sqrt(u_p)*u_q  
    # q     = 1 - np.sqrt(u_p)      

    logm  = logm_min + u_logm * (logm_max - logm_min)
    rs    = rs_min + u_rs * (rs_max - rs_min)

    pos_init_x = norm.ppf(u_pos_init_x, loc=mean_pos, scale=std_pos)
    pos_init_y = norm.ppf(u_pos_init_y, loc=mean_pos, scale=std_pos)
    pos_init_z = norm.ppf(u_pos_init_z, loc=mean_pos, scale=std_pos)

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel) 
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    x1 = u_x1
    x2 = u_x2
    x3 = u_x3

    # Return the transformed parameters
    return (logM, Rs, p, q,
            logm, rs,
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

def log_likelihood_MSE(params, dict_data, threshold=1.25):

    # Generate model predictions for the given parameters
    _, xy_model, _, _ = model(params)
    r_model     = np.sqrt(xy_model[0]**2 + xy_model[1]**2)
    theta_model = np.arctan2(xy_model[1], xy_model[0])
    r_model, theta_model, _ = unwrap(r_model, theta_model, None)

    f = interp1d(theta_model, r_model, kind='cubic', bounds_error=False, fill_value=np.nan)
    r_fit = f(dict_data['theta'])

    # Punish for too long
    dtheta_data  = abs(dict_data['theta'].max() - dict_data['theta'].min())
    dtheta_model = abs(theta_model.max() - theta_model.min())
    ratio = dtheta_model / dtheta_data
    too_long = ratio > threshold
    
    # Punish for too short
    arg_keep = ~np.isnan(r_fit)
    N_nan = np.sum(np.isnan(r_fit))

    logl = -0.5*np.sum( (dict_data['r'][arg_keep] - r_fit[arg_keep])**2/dict_data['r_sigma'][arg_keep]**2 + np.log(dict_data['r_sigma'][arg_keep]**2) ) - BAD_VAL *  N_nan - BAD_VAL * too_long * ratio

    return logl

def model(params, N_track=100, N_stars=500, Nbody=True, seed=True):
    # Unpack parameters
    logM, Rs, p, q,\
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end, \
    x1, x2, x3 = params

    rot_mat = get_rot_mat(x1, x2, x3)

    return Agama_stream(logM, Rs, p, q, 
                        logm, rs, 
                        pos_init_x, pos_init_y, pos_init_z, 
                        vel_init_x, vel_init_y, vel_init_z, 
                        t_end, rot_mat, 
                        N_track=N_track, N_stars=N_stars, Nbody=Nbody, seed=seed)
    
def get_data(ndim, N_data=20, seed=False, mean_p=None, mean_q=None, std_p=None, std_q=None, min_length=10, max_length=100, min_theta=np.pi/2, max_theta=5*np.pi/2):

    d_length = 0
    theta_length = 0
    theta_track_data = np.array([0])
    while d_length < min_length or theta_length < min_theta or max_length < d_length or max_theta < theta_length or np.any( np.diff(theta_track_data) < 0): 

        if seed:
            params_data = np.array([12, 15, 0.9, 0.8, 8, 2, -40, 0, 0, 0, 150, 0, 2, 0.5, 0.2, 0.7])

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

if __name__ == "__main__":
    # Hyperparameters
    ndim   = 16
    n_eff  = 10000
    nlive  = 100
    N_data = 20

    # Get Data
    dict_data, params_data = get_data(ndim, N_data, seed=True)

    
    # # Run and Save Dynesty
    # nworkers = os.cpu_count()
    # pool = Pool(nworkers)

    # sampler = dynesty.DynamicNestedSampler(log_likelihood_MSE,
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

    for i in range(10):
        params = params_data #prior_transform(np.random.rand(ndim))
        _, xy_track_model, _, _ = model(params, Nbody=True, seed=True)
        r_model     = np.sqrt(xy_track_model[0]**2 + xy_track_model[1]**2)
        theta_model = np.arctan2(xy_track_model[1], xy_track_model[0])
        r_model, theta_model, _ = unwrap(r_model, theta_model, None)
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1,2,1)
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
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

        plt.scatter(dict_data['x'], dict_data['y'], color='lime', label='Data')
        plt.plot(xy_track_model[0], xy_track_model[1], c='r', label='Model')
        plt.title(log_likelihood_MSE(params, dict_data))

        plt.subplot(1,2,2)
        plt.xlabel('Theta [rad]')
        plt.ylabel('r [kpc]')
        plt.plot(theta_model, r_model, c='r', label='Model')
        plt.errorbar(theta_data, dict_data['r'], yerr=dict_data['r_sigma'], fmt='o', color='lime', label='Data')
        plt.legend(loc='best')
        plt.show()





