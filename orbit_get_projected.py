import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import agama
# working units: 1 Msun, 1 kpc, 1 km/s
agama.setUnits(length=1, velocity=1, mass=1)

import gala.dynamics as gd
import gala.potential as gp
from astropy import units as auni

def model_orbit_agama(params, num_particles=int(1e2)):
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

def potential_spheroid(logM, Rs, q, dirx, diry, dirz):
    rot_mat = get_mat(dirx, diry, dirz)
    euler_angles = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    pot_host = agama.Potential(type='Spheroid', mass=10**logM, scaleRadius=Rs, axisRatioZ=q, orientation=euler_angles)
    return pot_host

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return R.from_rotvec(angle * v3).as_matrix()

def model_orbit_gala(params, n_steps=int(1e2)):
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
                                dt=t_end / n_steps * auni.Gyr,
                                n_steps=n_steps)
    xout, yout, _ = orbit.x.to_value(auni.kpc), orbit.y.to_value(
        auni.kpc), orbit.z.to_value(auni.kpc)
    
    xy_stream = np.array([xout, yout]).T

    return xy_stream