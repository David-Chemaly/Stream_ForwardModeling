import os
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import pickle

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
print('time unit: %.3f Gyr' % timeUnitGyr)

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

def model(params, num_particles=int(1e3), seed=42):
    # Unpack parameters
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    pos_init_x, pos_init_y, pos_init_z, \
    vel_init_x, vel_init_y, vel_init_z, \
    t_end = params

    pot_host = potential_spheroid(logM, Rs, q, dirx, diry, dirz)

    posvel_sat = np.array([pos_init_x, pos_init_y, pos_init_z,
                            vel_init_x, vel_init_y, vel_init_z])

    xy_stream = create_stream_particle_spray_with_progenitor(t_end, num_particles, pot_host, posvel_sat, 10**logm, rs, seed=seed, gala_modified=True)

    return xy_stream

def getData(seed=7, angle_bin=36, min_particule=10):

    rng = np.random.RandomState(seed)

    logM, Rs = 12., 15.
    q = 0.5
    dirx, diry, dirz = rng.normal(size=3)
    dirz = np.abs(dirz)

    logm, rs = 8., 1.

    x0, y0, z0    = rng.normal(size=3) * 100
    vx0, vy0, vz0 = rng.normal(size=3) * 100

    t_end = 3 # Gyr

    params = np.array([logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, t_end])

    xy_streams = model(params)

    x = xy_streams[:,0]
    y = xy_streams[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta[theta<0] += 2*np.pi

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
    r_sig = np.array(r_sig)
    theta_data = np.array(theta_data)
    x_data = r_data*np.cos(theta_data)
    y_data = r_data*np.sin(theta_data)

    data = {'r':r_data, 'theta':theta_data, 'r_sig':r_sig, 'x':x_data, 'y':y_data}

    return data, params

def prior_transform(p):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, y0, z0, vx0, vy0, vz0, \
    t_end = p

    logM1  = (11 + 2*logM)
    Rs1    = (5 + 20*Rs)
    q1     = q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz / 2]
    ]

    logm1 = (7 + 2*logm)
    rs1   = (0.5 + 1.5*rs)

    x1, y1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [x0, y0, z0]
    ]
    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx0, vy0, vz0]
    ]

    t_end1 = 4*t_end

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1, 
            x1, y1, z1, vx1, vy1, vz1, 
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
    delta_theta_data = np.diff(theta_data).min()/2
    r_sig = dict_data['r_sig']

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

if __name__ == '__main__':
    # Hyperparameters
    ndim  = 15
    nlive = 100
    seed  = 26
    dict_data, params = getData(seed=26, angle_bin=36, min_particule=10)
    DIR_SAVE = './test'

    # Run and Save Dynesty
    nworkers = os.cpu_count()
    print(nworkers)
    # pool = Pool(nworkers)
    dns      = dynesty.DynamicNestedSampler(log_likelihood,
                                            prior_transform, 
                                            sample='rslice',
                                            ndim=ndim, 
                                            nlive=nlive,
                                            bound='multi',
                                            # pool=pool, 
                                            # queue_size=nworkers, 
                                            logl_args=[dict_data])
    dns.run_nested(n_effective=10000)
    # pool.close()
    # pool.join()
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
    
    # Save dict_result as a pickle file
    with open(DIR_SAVE+'dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)

    with open(DIR_SAVE+'dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(DIR_SAVE+'params.txt', params)

