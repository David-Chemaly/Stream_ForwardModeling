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

BAD_VAL = -1e-100

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
    p     = 1 - np.sqrt(u_p)*u_q  
    q     = 1 - np.sqrt(u_p)      

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

def log_likelihood_MSE(params, dict_data):

    # Generate model predictions for the given parameters
    _, xy_model = model(params)
    r_model = np.sqrt(xy_model[0]**2 + xy_model[1]**2)
    theta_model = np.unwrap(np.arctan2(xy_model[1], xy_model[0]))

    f = interp1d(theta_model, r_model, kind='cubic', fill_value='extrapolate')
    r_fit = f(dict_data['theta'])

    # Compute the chi-squared
    logl = -np.sum((dict_data['r'] - r_fit)**2)/dict_data['sigma']**2

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
    
if __name__ == "__main__":
    # Hyperparameters
    ndim  = 16
    sigma = 2
    n_eff = 10000
    nlive = 100

    # Data
    params_data = np.array([12, 15, 0.9, 0.8, 8, 1, -40, 0, 0, 0, 150, 0, 2, 0.5, 0.2, 0.7])
    _, xy_track_data = model(params_data, N_stars=10000, Nbody=True, seed=False)

    r_track_data = np.sqrt(xy_track_data[0]**2 + xy_track_data[1]**2)
    r_track_data += np.random.normal(0, sigma, len(r_track_data))
    theta_track_data = np.unwrap(np.arctan2(xy_track_data[1], xy_track_data[0]))

    x_data = r_track_data * np.cos(theta_track_data)
    y_data = r_track_data * np.sin(theta_track_data)

    dict_data = {'theta':theta_track_data,
                 'r': r_track_data,
                 'sigma': sigma,
                 'x': x_data,
                 'y': y_data}
    
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
        params = prior_transform(np.random.rand(ndim))
        xy_stream_model, xy_track_model = model(params, Nbody=False)
    
        plt.figure(figsize=(10, 5))
        plt.scatter(dict_data['x'], dict_data['y'], color='b', label='Noisy Data track')
        plt.plot(xy_track_model[0], xy_track_model[1], c='r', label='Model track')
        plt.title(log_likelihood_MSE(params, dict_data))
        plt.legend(loc='best')
        plt.show()

        


