import os
import pickle
import random

import dynesty
import dynesty.utils as dyut
import multiprocessing as mp
from scipy.stats import truncnorm

import scipy
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as auni
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms

BAD_VAL = 1e50
VERY_BAD_VAL = -1e100

def prior_transform_stream(p):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, y0, z0, vx0, vy0, vz0, \
    t_end = p

    logM1  = (11 + 2*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, dirz]
    ]

    logm1 = (6 + 2*logm) 
    rs1   = (1 + 2*rs)

    x1, y1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [x0, y0, z0]
    ]

    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx0, vy0, vz0]
    ]

    t_end1 = 4 + t_end

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, y1, z1, vx1, vy1, vz1, 
            t_end1]

def get_data_stream(q_true, sigma=1, seed=42, n_ang=24, ndim=15):
    rng = np.random.RandomState(seed)
    correct = False

    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array(prior_transform_stream(p))
        params[2] = q_true

        r = (np.sum(params[8:11]**2))**.5
        units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]
        mat = get_mat(params[3], params[4], params[5])
        pot = gp.NFWPotential(10**params[0], params[1], 1, 1, params[2], R=mat, units=units)

        f_v = (np.sum(params[11:14]**2))**.5 / pot.circular_velocity(params[8:11]).item().value
        
        if (r > 20) & (r < 100) & (f_v > 0.4) & (f_v < 0.7):
            xy_stream = model_stream(params)
            
            dict_data = get_track_stream(xy_stream, n_ang=n_ang)
            if (len(dict_data['theta']) > n_ang//4):
                if (dict_data['r'].min() > 10) & (dict_data['r'].ptp() < 100) & (dict_data['theta'].min() > np.pi/4) & (dict_data['theta'].max() < 6*np.pi/4) & (np.diff( dict_data['theta'] ) < np.pi/4).all():
                    correct = True

    if sigma == 0:
        noise = 0
    else:
        noise = np.random.normal(0, sigma, len(dict_data['r']))
    dict_data['r'] += noise
    dict_data['r_sig'] = np.sqrt( dict_data['r_sig']**2 + sigma**2 )

    return dict_data, params

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()

def model_stream(params, dt=-10):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    w0 = gd.PhaseSpacePosition(
        pos=np.array([pos_init_x, pos_init_y, pos_init_z]) * auni.kpc,
        vel=np.array([vel_init_x, vel_init_y, vel_init_z]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    H = gp.Hamiltonian(pot)

    df = ms.FardalStreamDF(gala_modified=True, random_state=np.random.RandomState(42))

    prog_pot = gp.PlummerPotential(m=10**logm, b=rs, units=units)
    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, _ = gen.run(w0, 10**logm * auni.Msun, dt=dt* auni.Myr, n_steps=int(t_end * auni.Gyr/ abs(dt* auni.Myr)))
    xy_stream = stream.xyz.T[:, :2]

    return xy_stream.value

def get_track_stream(xy_stream, n_ang=24, min_star=10):
    NN = len(xy_stream)

    x = xy_stream[:, 0]
    y = xy_stream[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2*np.pi

    theta_bin = np.arange(0, 360, 360/n_ang) * np.pi/180

    theta_data = []
    r_data     = []
    sig_data   = []

    for i in range(n_ang-1):
        idx = np.where((theta >= theta_bin[i]) & (theta < theta_bin[i+1]))[0]

        if len(idx) > min_star:
            r_in = r[idx]

            theta_data.append((theta_bin[i] + theta_bin[i+1])/2)
            r_data.append(np.mean(r_in))
            sig_data.append(np.std(r_in))

    theta_data = np.array(theta_data)
    r_data = np.array(r_data)
    sig_data = np.array(sig_data)

    x_data = r_data * np.cos(theta_data)
    y_data = r_data * np.sin(theta_data)

    dict_data = {'theta': theta_data, 'r': r_data, 'x': x_data, 'y': y_data, 'r_sig': sig_data}

    return dict_data

def log_likelihood_stream(params, dict_data):
    xy_model   = model_stream(params)
    dict_model = get_track_stream(xy_model, n_ang=24)

    r_model = dict_model['r']
    theta_model = dict_model['theta']

    if (np.diff(theta_model) <= 0).any():
        logl = 2*VERY_BAD_VAL
    else:
        cs = CubicSpline(theta_model, r_model, extrapolate=False)

        r_data = dict_data['r']
        theta_data = dict_data['theta']
        r_sig = dict_data['r_sig']

        r_model = cs(theta_data)

        if (theta_data.min() >= theta_model.min() ) & (theta_data.max() <= theta_model.max()):
            logl = -.5 * np.sum( ( (cs(theta_data) - r_data) / r_sig )**2 )
        else:
            logl = VERY_BAD_VAL
            penalty  = max((np.maximum(theta_data, theta_model.min()) - theta_model.min())**2)
            penalty += max((np.minimum(theta_data, theta_model.max()) - theta_model.max())**2)
            logl = logl - np.abs(BAD_VAL) / 10000 * penalty 

    return logl


def dynesty_fit(dict_data, ndim=15, nlive=2500):
    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood_stream,
                                prior_transform_stream,
                                ndim,
                                logl_args=(dict_data, ),
                                nlive=nlive,
                                sample='rslice',  
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)
    
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }

def main(id, seed, q_true, ndim, nlive, sigma, dir_save):

    dict_data, params_data = get_data_stream(q_true, sigma=sigma, seed=seed, n_ang=24, ndim=ndim)
    print(log_likelihood_stream(params_data, dict_data))

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params_data)

    dict_result = dynesty_fit(dict_data, ndim, nlive)

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)

def run_in_parallel(q_mean, q_sig, seed, ndim, nlive, dir_save, sigma, N):
    # Create directory if it doesn't exist
    os.makedirs(dir_save, exist_ok=True)

    # Create a list of seeds, one for each process
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e5), N)

    print(f'Running {N} processes in parallel with {os.cpu_count()} cores')

    # Prepare arguments for each process
    for id, s in enumerate(seeds):
            if q_sig == 0:
                q_true = q_mean * np.zeros(N)
            else:
                trunc_gauss = truncnorm((0.5 - q_mean) / q_sig, (1.5 - q_mean) / q_sig, loc=q_mean, scale=q_sig)
                q_true = trunc_gauss.rvs(N, random_state=rng)

            main(id, s, q_true[id], ndim, nlive, sigma, dir_save)

if __name__ == '__main__':
    q_mean, q_sig, seed, ndim, nlive, sigma, N = 0.9, 0.1, 3, 12, 2500, 1, 100

    PATH_SAVE = f'/data/dc824-2/stream_to_stream/q{q_mean}_qsig{q_sig}_seed{seed}_nlive{nlive}_sigma{sigma}'
    run_in_parallel(q_mean, q_sig, seed, ndim, nlive, PATH_SAVE, sigma, N)