import scipy
from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as auni
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms

def prior_transform_data(p):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, y0, z0, vx0, vy0, vz0, \
    t_end = p

    logM1  = (11 + 2*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.4 + 0.6*q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, dirz]
    ]

    logm1 = (6 + 2*logm) 
    rs1   = (1 + 2*rs)

    x1, y1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [x0, y0, z0]
    ]
    # x1 = -abs(x1)
    # y1 = 0

    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx0, vy0, vz0]
    ]
    # vx1 = 0
    # vz1 = 0

    t_end1 = 4 + t_end

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, y1, z1, vx1, vy1, vz1, 
            t_end1]

def get_data_stream(q_true, seed=42, n_ang=18, ndim=15):
    rng = np.random.RandomState(seed)
    correct = False

    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array(prior_transform_data(p))
        params[2] = q_true

        r = (np.sum(params[8:11]**2))**.5
        units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]
        mat = get_mat(params[3], params[4], params[5])
        pot = gp.NFWPotential(10**params[0], params[1], 1, 1, params[2], R=mat, units=units)

        f_v = (np.sum(params[11:14]**2))**.5 / pot.circular_velocity(params[8:11]).item().value

        # logM  = 11 + 2 * rng.uniform(0, 1) 
        # Rs    =  5 + 20 * rng.uniform(0, 1)
        # q     = q_true
        # dirx, diry, dirz = [rng.uniform(0, 1) for _ in range(3)]

        # logm, rs = 7, 1
        # r0 = 20 + 60 * rng.uniform(0, 1)

        # units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]
        # mat = get_mat(dirx, diry, dirz)
        # pot = gp.NFWPotential(10**logM, Rs, 1, 1, q, R=mat, units=units)

        # v_c = pot.circular_velocity([-r0, 0, 0]).item().value

        # params = np.array([logM, Rs, q, dirx, diry, dirz, logm, rs, -r0, 0, 0, 0, v_c, 0, 4])
        
        if (r > 20) & (r < 100) & (f_v > 0.4) & (f_v < 0.7):
            xy_stream = model_stream(params)
            
            dict_data = get_track(xy_stream, n_ang=n_ang)
            if (len(dict_data['theta']) > n_ang//4):
                if (dict_data['theta'].min() > np.pi/4) & (dict_data['theta'].max() < 6*np.pi/4) & (np.diff( dict_data['theta'] ) < 1).all():
                    correct = True

    return dict_data, params

def get_track(xy_stream, n_ang=144, min_star=10):
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

    df = ms.FardalStreamDF(gala_modified=True)

    prog_pot = gp.PlummerPotential(m=10**logm, b=rs, units=units)
    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, _ = gen.run(w0, 10**logm * auni.Msun, dt=dt* auni.Myr, n_steps=int(t_end * auni.Gyr/ abs(dt* auni.Myr)))
    xy_stream = stream.xyz.T[:, :2]

    return xy_stream.value

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()