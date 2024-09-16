import os
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import dynesty
import dynesty.utils as dyut
from multiprocessing import Pool

import scipy
from scipy.spatial.transform import Rotation as R
from sklearn.mixture import GaussianMixture

import agama
# working units: 1 Msun, 1 kpc, 1 km/s
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)

from astropy.constants import G
from astropy import units as u
G = G.to(u.kpc*(u.km/u.s)**2/u.Msun)

BAD_VAL = 1e50

# Get rot matrix
def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()

def potential_spheroid(logM, Rs, q, dirx, diry, dirz):
    rot_mat = get_mat(dirx, diry, dirz)
    euler_angles = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    pot_host = agama.Potential(type='Spheroid', mass=10**logM, scaleRadius=Rs, axisRatioZ=q, orientation=euler_angles)
    return pot_host

def model(params, num_particles=int(1e4)):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    pos_init_x, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    pot_host = potential_spheroid(logM, Rs, q, dirx, diry, dirz)

    posvel_sat = np.array([pos_init_x, 0., pos_init_z,
                            vel_init_x, vel_init_y, vel_init_z])

    xv_stream = agama.orbit(potential=pot_host, ic=posvel_sat, time=-t_end, trajsize=num_particles, verbose=False)
    xy_stream = xv_stream[1][:, :2]

    return xy_stream

def restrictions(dtheta, dtheta_min = np.pi/2, dtheta_max = 2*np.pi):
    if (dtheta_min < dtheta) & (dtheta < dtheta_max):
        return True
    else:
        return False

def getData_orbit(q_true, ndim, seed=7, sigma=1, angle_bin=18, min_particule=10):

    rng = np.random.RandomState(seed)

    correct = False
    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array( prior_transform(p) )
        params[2] = q_true

        xy_streams = model(params)

        x = xy_streams[:,0]
        y = xy_streams[:,1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta<0] += 2*np.pi

        correct = restrictions(theta.ptp())


    theta_bin = np.linspace(0, 2*np.pi, angle_bin)

    theta_data = []
    r_data = []
    r_sig = []
    for i in range(len(theta_bin)-1):
        idx = np.where((theta>theta_bin[i]) & (theta<theta_bin[i+1]))[0]

        if len(idx) > min_particule:
            gmm1 = GaussianMixture(n_components=1, covariance_type='full').fit(r[idx].reshape(-1,1))

            r_data.append(gmm1.means_[0][0])
            r_sig.append(np.sqrt(gmm1.covariances_[0][0][0]))
            theta_data.append((theta_bin[i]+theta_bin[i+1])/2.)
    r_data = np.array(r_data) 
    if sigma !=0:
        r_data += rng.normal(0, sigma, size=len(r_data))
    r_sig  = np.sqrt( np.array(r_sig)**2 + sigma**2)
    theta_data = np.array(theta_data)
    x_data = r_data*np.cos(theta_data)
    y_data = r_data*np.sin(theta_data)

    data = {'r':r_data, 'theta':theta_data, 'r_sig':r_sig, 'x':x_data, 'y':y_data, 'dtheta': (theta_bin[1]-theta_bin[0])/2}
    print(log_likelihood(params, data))

    return data, params

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

    t_end1 = 1.5*t_end

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            x1, z1, vx1, vy1, vz1, 
            t_end1]

def log_likelihood(params, dict_data):
    xy_streams = model(params)

    x_model = xy_streams[:,0]
    y_model = xy_streams[:,1]
    r_model = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.arctan2(y_model, x_model)
    theta_model[theta_model<0] += 2*np.pi

    r_data = dict_data['r']
    theta_data = dict_data['theta']
    delta_theta_data = dict_data['dtheta'] #np.diff(theta_data).min()/2
    r_sig = dict_data['r_sig']

    if not restrictions(theta_model.ptp()):
        logl = -BAD_VAL * 100 

    else:
        logl = 0
        for i in np.unique(theta_data):
            idx_data  = np.where(theta_data==i)[0] 
            idx_model = np.where( (theta_model>i-delta_theta_data) & (theta_model<i+delta_theta_data))[0]

            if len(idx_model) >= 10:
                gmm_fit = GaussianMixture(n_components=len(idx_data), covariance_type='full', random_state=42).fit(r_model[idx_model].reshape(-1,1))
                r_fit = gmm_fit.means_.flatten()

                logl += -0.5*np.sum( (np.sort(r_data[idx_data]) - np.sort(r_fit))**2/r_sig[idx_data]**2 + np.log(r_sig[idx_data]**2) )        
            else:
                logl += -BAD_VAL

    return logl
    
def main(id, seed, q_true, ndim, nlive, sigma, dir_save):
    dict_data, params = getData_orbit(q_true, ndim, seed=seed, sigma=sigma, angle_bin=18, min_particule=10)

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params)

    dns      = dynesty.DynamicNestedSampler(log_likelihood,
                                            prior_transform, 
                                            sample='rslice',
                                            ndim=ndim, 
                                            nlive=nlive,
                                            bound='multi',
                                            logl_args=[dict_data])
    dns.run_nested(n_effective=10000)
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl = res.logl[inds]

    dict_result = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)


def run_in_parallel(q_true, seed, ndim, nlive, dir_save, sigma, N):
    # Create directory if it doesn't exist
    os.makedirs(dir_save, exist_ok=True)

    # Create a list of seeds, one for each process
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e5), N)

    # Prepare arguments for each process
    args = [(id, s, q_true, ndim, nlive, sigma, dir_save) for id, s in enumerate(seeds)]

    # Use multiprocessing Pool to run the function in parallel
    print(f'Running {N} processes in parallel with {os.cpu_count()} cores')
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(main, args)

if __name__ == '__main__':
    q_true = 0.75
    seed  = 42
    sigma = 1
    ndim  = 12
    nlive = 1200
    dir_save = f'/data/dc824-2/orbit_to_orbit_ME/q{q_true}_seed{seed}_nlive{nlive}'
    N = 1
    
    run_in_parallel(q_true, seed, ndim, nlive, dir_save, sigma, N)