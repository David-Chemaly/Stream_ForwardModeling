import os
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import h5py
import time
import pickle

import dynesty
import dynesty.utils as dyut
from concurrent.futures import ProcessPoolExecutor, as_completed

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

BAD_VAL       = 1e50
SUPER_BAD_VAL = 1e80

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * np.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm

### Prior transform function ###
def prior_transform(utheta):
    # Unpack the unit cube values
    u_logM, u_Rs, u_gamma, u_beta, \
    u_pos_init_x, u_pos_init_z, \
    u_vel_init_x, u_vel_init_y, u_vel_init_z, \
    u_t_end = utheta

    logM_min, logM_max   = 11, 13
    Rs_min, Rs_max       = 5, 25
    gamma_min, gamma_max = 0, 2
    beta_min, beta_max   = 2, 4

    mean_pos = 0
    std_pos  = 100

    mean_vel = 0
    std_vel  = 100

    t_end_min, t_end_max = 1, 3
    
    # Transform each parameter
    logM  = logM_min + u_logM * (logM_max - logM_min) 
    Rs    = Rs_min + u_Rs * (Rs_max - Rs_min)  
    gamma = gamma_min + u_gamma * (gamma_max - gamma_min)
    beta  = beta_min + u_beta * (beta_max - beta_min)

    pos_init_x = abs(norm.ppf(u_pos_init_x, loc=mean_pos, scale=std_pos))
    pos_init_z = norm.ppf(u_pos_init_z, loc=mean_pos, scale=std_pos)

    vel_init_x = norm.ppf(u_vel_init_x, loc=mean_vel, scale=std_vel)
    vel_init_y = abs(norm.ppf(u_vel_init_y, loc=mean_vel, scale=std_vel))
    vel_init_z = norm.ppf(u_vel_init_z, loc=mean_vel, scale=std_vel)

    t_end = t_end_min + u_t_end * (t_end_max - t_end_min)

    # Return the transformed parameters
    return (logM, Rs, gamma, beta,
            pos_init_x, pos_init_z, 
            vel_init_x, vel_init_y, vel_init_z,
            t_end) 

def model(params, trajsize=1001):

    # Unpack parameters
    logM, Rs, gamma, beta, \
    pos_init_x, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, 1, 1)

    # Set host potential
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=gamma, beta=beta, 
                                axisRatioY=1, axisRatioZ=1)

    # initial displacement
    r_center = np.array([pos_init_x, 0, pos_init_z, vel_init_x, vel_init_y, vel_init_z])

    _, xv = agama.orbit(ic=r_center, potential=pot_host, time=t_end, timestart=0, trajsize=trajsize, verbose=False)

    xy_orbit = xv[:, :2].T

    return xy_orbit

def restriction(x, y, r, theta, d_min=10, r_min=10, theta_min=np.pi/2, theta_max=6*np.pi/2, R_threshold=0.9):

    # Check if orbit is long enough
    arc_lengths  = abs(r[:-1] * np.diff(theta))
    cumulative_arc_lengths = np.concatenate([[0], np.cumsum(arc_lengths)])
    d_length     = cumulative_arc_lengths[-1]

    # Check if orbit is curved enough
    theta_length = abs(theta.max() - theta.min())

    # Check if orbit is not just a line
    _, _, r_value, _, _ = linregress(x, y)

    # Check if orbit is coming back on itself
    overlap = (np.diff(theta) <= 0).sum()

    if d_length > d_min and theta_length > theta_min and theta_length < theta_max and r.min() > r_min and abs(r_value) < R_threshold and overlap == 0:
        return True, None
    else:
        if d_length < d_min:
            d_ratio = 1 + (d_min-d_length)/d_min
        else:
            d_ratio = 1
        if theta_length < theta_min:
            theta_ratio_min = 1 + (theta_min-theta_length)/theta_min
        else:
            theta_ratio_min = 1
        if theta_length > theta_max:
            theta_ratio_max = 1 + (theta_length-theta_max)/theta_length
        else:
            theta_ratio_max = 1
        if r.min() < r_min:
            r_ratio = 1 + (r_min-r.min())/r_min
        else:
            r_ratio = 1
        if abs(r_value) > R_threshold:
            R_ratio = 1 + (abs(r_value)-R_threshold)/abs(r_value)
        else:
            R_ratio = 1
        if overlap != 0:
            overlap_ratio = 1 + overlap/len(theta)
        else:
            overlap_ratio = 1
        punishement = d_ratio + theta_ratio_min + theta_ratio_max + r_ratio + R_ratio + overlap_ratio
        return False, punishement


def log_likelihood(params, dict_data):

    # Generate model predictions for the given parameters
    x, y = model(params)
    r_model  = np.sqrt(x**2 + y**2)
    theta_model = np.unwrap(np.arctan2(y, x))

    IsGood, punish = restriction(x, y, r_model, theta_model)
    if not IsGood:
        logl = -SUPER_BAD_VAL * punish

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

def getData(ndim, gamma=1, beta=3, sigma=3, N_data=20):

    IsGood = False
    while not IsGood:
        p = np.random.uniform(0, 1, ndim)
        params = np.array(prior_transform(p))
        params[2] = gamma
        params[3] = beta

        x, y = model(params)
        r = np.sqrt(x**2 + y**2)
        theta = np.unwrap(np.arctan2(y, x))

        IsGood = restriction(x, y, r, theta)

    theta_length = abs(theta.max() - theta.min())
    dtheta       = theta_length/N_data  # kpc -- Desired fixed distance
    theta_data   = np.arange(dtheta, theta_length-dtheta, dtheta)

    f = interp1d(theta, r, kind='cubic', bounds_error=False, fill_value=np.nan)
    r_data = f(theta_data)

    noise = np.random.normal(0, sigma, len(r_data))
    r_data = r_data + noise

    x_data = r_data * np.cos(theta_data)
    y_data = r_data * np.sin(theta_data)

    dict_data = {'theta': theta_data,
                'r': r_data,
                'r_sigma': np.zeros(len(r_data))+noise,
                'x': x_data,
                'y': y_data}

    return dict_data, params

def fit_one(path_save, ndim, nlive, gamma, beta, sigma, N_data, index):

    save_stream = f'{path_save}/xx_{index+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    dict_data, params = getData(ndim, gamma, beta, sigma=sigma, N_data=N_data)
    np.savetxt(f'{save_stream}/true_params.txt', params)

    sampler, kw = 'rwalk', {'walks': None}
    sampler, kw = 'rslice', {}

    dns = dynesty.DynamicNestedSampler(log_likelihood,
                                    prior_transform,
                                    ndim,
                                    logl_args=(dict_data, ),
                                    nlive=nlive,
                                    sample=sampler,
                                    **kw)
    dns.run_nested(n_effective=10000)

    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl = res.logl[inds]

    res = {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }

    with open(f'{save_stream}/dict_dns.pkl', 'wb') as file:
                pickle.dump(res, file)


if __name__ == "__main__":
    # Path to save
    gamma, beta = 1, 3
    # PATH_SAVE = f'/data/dc824-2/orbit_to_orbit_fit_0D_Denis_AllME/gamma{gamma}_beta{beta}'
    PATH_SAVE = f'./gamma{gamma}_beta{beta}'

    # Hyperparameters
    ndim   = 10
    nlive  = 1200
    sigma  = 3
    N_data = 20

    N_orbits = 1

    nworkers = os.cpu_count()
    print(f'Number of workers: {nworkers}')

    with Pool(nworkers) as poo:
        args = [(PATH_SAVE, ndim, nlive, gamma, beta, sigma, N_data, i) for i in range(N_orbits)]
        poo.starmap(fit_one, args)
