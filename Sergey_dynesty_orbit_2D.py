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

BAD_VAL = 1e50
SUPER_BAD_VAL = 1e80

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * np.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm

# Get rot matrix
def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()
    
### Prior transform function ###
def prior_transform(utheta):
    # Unpack the unit cube values
    u_logM, u_Rs, u_p, \
    u_pos_init_x, u_pos_init_y, u_pos_init_z, \
    u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end, \
    u_dirx, u_diry, u_dirz = utheta

    logM_min, logM_max = 11, 13
    Rs_min, Rs_max     = 5, 25
    p_min, p_max       = 0.4, 1.

    mean_pos = 0
    std_pos  = 100

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  
    p     = p_min + u_p * (p_max - p_min) 

    pos_init_x = abs(norm.ppf(u_pos_init_x, loc=mean_pos, scale=std_pos))
    pos_init_y = abs(norm.ppf(u_pos_init_y, loc=mean_pos, scale=std_pos/100))
    pos_init_z = norm.ppf(u_pos_init_z, loc=mean_pos, scale=std_pos)

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = abs(norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel))
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    dirx = norm.ppf(u_dirx, loc=0, scale=1)
    diry = norm.ppf(u_diry, loc=0, scale=1)
    dirz = abs(norm.ppf(u_dirz, loc=0, scale=1))

    # Return the transformed parameters
    return (logM, Rs, p,
            pos_init_x, pos_init_y, pos_init_z, 
            vel_init_x, vel_init_y, vel_init_z,
            t_end, 
            dirx, diry, dirz)
    
def log_likelihood_MSE(params, dict_data):

    # Generate model predictions for the given parameters
    xy_model    = model(params)
    r_model     = np.sqrt(xy_model[0]**2 + xy_model[1]**2)
    theta_model = np.unwrap(np.arctan2(xy_model[1], xy_model[0]))

    overlap = (np.diff(theta_model) <= 0).sum()
    if overlap != 0:
        logl = -SUPER_BAD_VAL * overlap

    # else: 
    #     f = interp1d(theta_model, r_model, kind='cubic', bounds_error=False, fill_value=np.nan)
    #     r_fit = f(dict_data['theta'])

    #     # Punish for too short
    #     N_nan = np.sum(np.isnan(r_fit))

    #     if N_nan > 0:
    #         logl = -BAD_VAL *  N_nan
    #     else:
    #         logl = -0.5*np.sum( (dict_data['r'] - r_fit)**2/dict_data['r_sigma']**2 + np.log(dict_data['r_sigma']**2) )

    else:
        min_m_ang = theta_model.min()
        max_m_ang = theta_model.max()
        d_ang = (dict_data['theta'] + 10 * np.pi - min_m_ang) % (2 * np.pi) + min_m_ang
        # min_model min_data max_data max_model
        # how it should be
        if (theta_model.min() < dict_data['theta'].min()) & (theta_model.max() > dict_data['theta'].max()):
            f = interp1d(theta_model, r_model, kind='cubic', bounds_error=False, fill_value=np.nan)
            r_fit = f(dict_data['theta'])

            logl = -0.5*np.sum( (dict_data['r'] - r_fit)**2/dict_data['r_sigma']**2 + np.log(dict_data['r_sigma']**2) )

        else:
            logl = -BAD_VAL
            # penalty = max((np.maximum(d_ang, min_m_ang) - min_m_ang)**2)
            # penalty += max((np.minimum(d_ang, max_m_ang) - max_m_ang)**2)
            # logl = logl - np.abs(BAD_VAL) / 10000 * penalty

        # theta_model_min = theta_model.min()
        # theta_model_max = theta_model.max()
        # d_ang = (dict_data['theta'] + 10 * np.pi - theta_model_min) % (2 * np.pi) + theta_model_min

        # if ((d_ang > theta_model_min) & (d_ang < theta_model_max)).all():
        #     #penalty = max(d_ang.min() - theta_model_min - 0.1, 0)**2 + max(theta_model_max - d_ang.max() - 0.1, 0)**2
        #     II      = CubicSpline(theta_model, r_model)
        #     f = interp1d(theta_model, r_model, kind='cubic', bounds_error=False, fill_value=np.nan)
        #     logl = -0.5*np.sum( (dict_data['r'] - f(d_ang))**2/dict_data['r_sigma']**2)# + np.log(dict_data['r_sigma']**2) )

        #     # logl  = -.5 * np.sum(((f(dict_data['theta']) - dict_data['r']) / dict_data['r_sigma'])**2) #- 10 * penalty

        # else:
        #     logl = BAD_VAL
        #     # penalty = max((np.maximum(d_ang, theta_model_min) - theta_model_min)**2)
        #     # penalty += max((np.minimum(d_ang, theta_model_max) - theta_model_max)**2)
        #     # logl = logl - np.abs(BAD_VAL) / 10000 * penalty
            
    return logl

def model(params, dt=0.001):
    # Unpack parameters
    logM, Rs, p, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end, \
    dirx, diry, dirz = params

    rot_mat = get_mat(dirx, diry, dirz)
    rot     = R.from_matrix(rot_mat)
    euler_angles = rot.as_euler('xyz', degrees=False)

    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, 1, p)

    # Set host potential
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, 
                                axisRatioY=1, axisRatioZ=p, orientation=euler_angles)

    # initial displacement
    r_center = np.array([pos_init_x, pos_init_y, pos_init_z, vel_init_x, vel_init_y, vel_init_z])

    _, xv = agama.orbit(ic=r_center, potential=pot_host, time=t_end, timestart=0, trajsize=int(t_end/dt)+1, verbose=False)

    xy_orbit = xv[:, :2].T

    return xy_orbit

def get_data_orbit(ndim, p_flat, step_type='distance', sigma=1, N_data=20, d_min=30, theta_min=np.pi/2, theta_max=5*np.pi/2, r_min=10, R_threshold=0.9):
    not_yet = True
    while not_yet:
        p = np.random.uniform(0, 1, ndim)
        params = np.array(prior_transform(p))
        params[2] = p_flat

        x, y = model(params)
        r = np.sqrt(x**2 + y**2)
        theta = np.unwrap(np.arctan2(y, x))

        overlap = (np.diff(theta) <= 0).sum()

        if overlap == 0:

            arc_lengths  = abs(r[:-1] * np.diff(theta))
            cumulative_arc_lengths = np.concatenate([[0], np.cumsum(arc_lengths)])
            d_length     = cumulative_arc_lengths[-1]
            theta_length = abs(theta.max() - theta.min())

            _, _, r_value, _, _ = linregress(x, y)

            if d_length > d_min and theta_length > theta_min and theta_length < theta_max and r.min() > r_min and abs(r_value) < R_threshold:
                not_yet = False

    if step_type == 'distance':
        d = d_length/N_data  # kpc -- Desired fixed distance
        d_fit = np.arange(d, d_length-d, d)

        f = interp1d(cumulative_arc_lengths, theta, kind='cubic')
        theta_fit = f(d_fit)

        f = interp1d(theta, r, kind='cubic')
        r_fit = f(theta_fit)

    elif step_type == 'theta':
        d = theta_length/N_data  # kpc -- Desired fixed distance
        theta_fit = np.arange(d, theta_length-d, d)

        f = interp1d(theta, r, kind='cubic')
        r_fit = f(theta_fit)

    noise = np.random.normal(0, sigma, len(r_fit))
    r_fit = r_fit + noise

    x_fit = r_fit * np.cos(theta_fit)
    y_fit = r_fit * np.sin(theta_fit)

    dict_data = {'theta': theta_fit,
                'r': r_fit,
                'r_sigma': np.zeros(len(r_fit))+sigma,
                'x': x_fit,
                'y': y_fit}
    
    return dict_data, params

if __name__ == "__main__":
    # Path to save
    p_flat = 0.8
    PATH_SAVE = f'/data/dc824-2/orbit_to_orbit_fit_2D_Sergey/p{p_flat}'

    # Hyperparameters
    ndim   = 13
    nlive  = 1200
    N_data = 20
    sigma  = 3

    # Save params
    save_directory = f'{PATH_SAVE}/Stream_1' 

    # Get Data
    dict_data, params_data = get_data_orbit(ndim, p_flat, 'distance', sigma=sigma, N_data=N_data)

    # Run and Save Dynesty
    nworkers = os.cpu_count()
    print(nworkers)
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
