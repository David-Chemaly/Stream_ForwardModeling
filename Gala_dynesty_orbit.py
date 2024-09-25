import os
import scipy
import pickle
import numpy as np
import multiprocessing as mp
from scipy.stats import truncnorm

from astropy import units as auni

import gala.dynamics as gd
import gala.potential as gp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

import dynesty
import dynesty.utils as dyut

BAD_VAL = 1e50
VERY_BAD_VAL = 1e100

def prior_transform(p):
    logM, Rs, q, dirx, diry, dirz, \
    x0, z0, vx0, vy0, vz0, \
    t_end = p

    logM1  = (11 + 2*logM)
    Rs1    = (5 + 20*Rs)
    q1     = q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    x1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [0.5 + x0/2, z0]
    ]
    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx0, 0.5 + vy0/2, vz0]
    ]

    # t_end1 = 2*t_end
    t_end1 = 10**(2 + t_end)

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            x1, z1, vx1, vy1, vz1, 
            t_end1]

def log_likelihood(params, dict_data):
    xy_model = model(params)
    x_model = xy_model[:,0]
    y_model = xy_model[:,1]
    r_model = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.unwrap( np.arctan2(y_model, x_model) )

    if (np.diff(theta_model) <= 0).any() or theta_model.ptp() > 2*np.pi:
        logl = -VERY_BAD_VAL
    else:
        cs = CubicSpline(theta_model, r_model, extrapolate=False)

        r_data = dict_data['r']
        theta_data = dict_data['theta']
        r_sig = dict_data['r_sig']

        r_model = cs(theta_data)

        if np.isnan(r_model).any():
            logl = -BAD_VAL * np.sum(np.isnan(r_model))

        else:
            delta_r = r_data - r_model
            logl    = -0.5 * np.sum(delta_r**2 / r_sig**2 )

    return logl

def model(params, n_steps=int(1e3)):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    pos_init_x, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]

    w0 = gd.PhaseSpacePosition(
        pos=np.array([pos_init_x, 0, pos_init_z]) * auni.kpc,
        vel=np.array([vel_init_x, vel_init_y, vel_init_z]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

    orbit = pot.integrate_orbit(w0,
                                dt=t_end / n_steps * auni.Myr, # auni.Gyr
                                n_steps=n_steps)
    xout, yout, _ = orbit.x.to_value(auni.kpc), orbit.y.to_value(
        auni.kpc), orbit.z.to_value(auni.kpc)
    
    xy_stream = np.array([xout, yout]).T

    return xy_stream

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()

def get_data(q_true, ndim, seed=42, sigma=1, n_ang=36):
    rng = np.random.RandomState(seed)
    correct = False

    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array( prior_transform(p) )
        params[2] = q_true

        xy = model(params)
        x = xy[:,0]
        y = xy[:,1]
        r = (x**2 + y**2)**.5
        theta = np.unwrap( np.arctan2(y, x) )

        if (np.diff(theta) > 0).all() & (theta.ptp() < 2*np.pi) & (theta.ptp() > np.pi/2): 
            correct = True

    cs = CubicSpline(theta, r, extrapolate=False)

    theta_bin = np.arange(0, 360, 360/n_ang) * np.pi/180
    r_bin = cs(theta_bin)

    arg_in = ~np.isnan(r_bin)

    theta_data = theta_bin[arg_in]
    r_data     = r_bin[arg_in]
    x_data = r_data * np.cos(theta_data)
    y_data = r_data * np.sin(theta_data)
    
    if sigma == 0:
        noise = 0
    else:
        noise = np.random.normal(0, sigma, len(r_data))

    dict_data = {'theta': theta_data, 'r': r_data+noise, 'x': x_data, 'y': y_data, 'r_sig': sigma}

    return dict_data, params

def dynesty_fit(dict_data, ndim=12, nlive=1200):
    
    dns = dynesty.DynamicNestedSampler(log_likelihood,
                                    prior_transform,
                                    ndim,
                                    logl_args=(dict_data, ),
                                    nlive=nlive,
                                    sample='rslice')
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

    dict_data, params_data = get_data(q_true, ndim, seed, sigma, n_ang=36)
    print(log_likelihood(params_data, dict_data))

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params_data)

    dict_result = dynesty_fit(dict_data, ndim, nlive)

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)

def run_in_parallel(q_true, seed, ndim, nlive, dir_save, sigma, N):
    # Create directory if it doesn't exist
    os.makedirs(dir_save, exist_ok=True)

    # Create a list of seeds, one for each process
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e5), N)

    # Prepare arguments for each process
    args = [(id, s, q_true, ndim, nlive, sigma, dir_save) for id, s in enumerate(seeds)]

    # Use multiprocessing Pool to run the function in parallel
    print(f'Running {N} processes in parallel with {os.cpu_count()} cores')
    with mp.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(main, args)
