import os, sys, time as timer
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import contextlib, random, numpy as np, agama, scipy.special, scipy.integrate, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import LSQUnivariateSpline
import pyfalcon
import time as timer
from astropy.constants import G
from astropy import units as u
G = G.to(u.kpc*(u.km/u.s)**2/u.Msun).value

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * np.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm

def orbitDF(pot, ic, time, timestart, trajsize, mass):
    # integrate the orbit of a massive particle in the host galaxy, accounting for dynamical friction
    if mass == 0:
        return agama.orbit(ic=ic, potential=pot, time=time, timestart=timestart, trajsize=trajsize, verbose=False)
    times = np.linspace(timestart, timestart+time, trajsize)
    traj = scipy.integrate.odeint(
        lambda xv, t: np.hstack((xv[3:6], pot.force(xv[0:3], t=t))),
        ic, times)
    return times, traj

def Agama_stream(logM, Rs, p, q, logm, rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, N_track=100, N_stars=500, Nbody=False, seed=True, degree=5):
    # working units: 1 Msun, 1 kpc, 1 km/s
    agama.setUnits(length=1, velocity=1, mass=1)
    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, p, q)

    # Set host potential
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, axisRatioY=p, axisRatioZ=q)


    # Set satellite potential
    pot_sat  = agama.Potential(type='Plummer', mass=10**logm, scaleradius=rs)
    initmass = pot_sat.totalMass()

    # create a spherical isotropic DF for the satellite and sample it with particles   
    if seed == True: 
        xv = np.load('xv0.npy')
        xv[:, :3] *= rs
        xv[:, 3:] *= np.sqrt(10**logm/rs)
        mass = np.ones(N_stars) * 10**logm/N_stars
    else:
        df_sat = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
        xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(N_stars)


    # initial displacement
    r_center = np.array([x0, y0, z0, vx0, vy0, vz0])
    xv += r_center

    # parameters for the simulation
    tupd = 2**-2 # interval for plotting and updating the satellite mass for the restricted N-body simulation
    tau  = 2**-8 # timestep of the full N-body sim (typically should be smaller than eps/v, where v is characteristic internal velocity)
    eps  = 0.1   # softening length for the full N-body simulation


    # simulate the evolution of the disrupting satellite using two methods:
    # "restricted N-body" (r_ prefix) and "full N-body" (if available, f_ prefix)
    r_mass   = [initmass]
    r_xv     = xv.copy()
    time     = 0.0   # current simulation time

    f_gamma  = (r_xv[:, 0:3]**2).sum(axis=1) - (np.array(r_center.copy()[0:3])**2).sum()

    while time < tend:

        time_center, orbit_center = orbitDF(pot=pot_host, ic=r_center, time=tupd, timestart=time, trajsize=round(tupd/tau) + 1, mass=r_mass[-1])
        r_center = orbit_center[-1]  # current position and velocity of satellite CoM

        if Nbody == False:

            # initialize the time-dependent total potential (host + moving sat) on this time interval
            pot_total = agama.Potential(pot_host,
                agama.Potential(potential=pot_sat, center=np.column_stack((time_center, orbit_center))))
            # compute the trajectories of all particles moving in the combined potential of the host galaxy and the moving satellite
            r_xv = np.vstack(agama.orbit(ic=r_xv, potential=pot_total, time=tupd, timestart=time, trajsize=1, verbose=False)[:,1])
            f_gamma += (r_xv[:, 0:3]**2).sum(axis=1) - (np.array(r_center.copy()[0:3])**2).sum()

        if Nbody == True:
            if time==0:   # initialize accelerations and potential
                f_acc, f_pot = pyfalcon.gravity(r_xv[:,0:3], agama.G * mass, eps)
                f_acc += pot_host.force(r_xv[:,0:3])

            f_time = 0
            while f_time < tupd:
                # kick-drift-kick leapfrog method:
                # kick for half-step, using accelerations computed at the end of the previous step
                r_xv[:,3:6] += f_acc * (tau/2)
                # drift for full step
                r_xv[:,0:3] += r_xv[:,3:6] * tau
                # recompute accelerations from self-gravity of the satellite
                # NB: falcON works with natural N-body units in which G=1, so we multiply particle mass passed to falcon by G
                f_acc, _ = pyfalcon.gravity(r_xv[:,0:3], agama.G * mass, eps)
                # add accelerations from the host galaxy
                f_acc += pot_host.force(r_xv[:,0:3])
                # kick again for half-step
                r_xv[:,3:6] += f_acc * (tau/2)
                f_time += tau
            f_gamma += ((r_xv[:, 0:3]**2).sum(axis=1)) - ((np.array(r_center.copy()[0:3])**2).sum())
                
        time += tupd

    lower_bound = np.percentile(f_gamma, 5)
    upper_bound = np.percentile(f_gamma, 95)

    arg_keep = np.where( (f_gamma >= lower_bound) & (f_gamma <= upper_bound))[0]
    gamma    = f_gamma[arg_keep]
    arg_sort = np.argsort(gamma)
    gamma    = gamma[arg_sort]

    xy_stream  = (rot_mat @  r_xv[arg_keep, 0:3].T)[:2, arg_sort]

    try:
        x_spline = LSQUnivariateSpline(gamma, xy_stream[0], t=[0], k=degree)
        y_spline = LSQUnivariateSpline(gamma, xy_stream[1], t=[0], k=degree)
    except:
        x_spline = LSQUnivariateSpline(gamma, xy_stream[0], t=[gamma[len(gamma)//2]], k=degree)
        y_spline = LSQUnivariateSpline(gamma, xy_stream[1], t=[gamma[len(gamma)//2]], k=degree)

    gamma_fit    = np.linspace(gamma.min(), gamma.max(), N_track)
    xy_track     = np.zeros((2, N_track))
    xy_track[0]  = x_spline(gamma_fit)
    xy_track[1]  = y_spline(gamma_fit)

    return xy_stream, xy_track, gamma, gamma_fit

def scale_to_unit_interval(x):
    return (2 * x - (x.max() + x.min())) / (x.max() - x.min())

def Agama_orbit(logM, Rs, p, q, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, N_orbit=1000):
    # working units: 1 Msun, 1 kpc, 1 km/s
    agama.setUnits(length=1, velocity=1, mass=1)
    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, p, q)

    # Set host potential
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, axisRatioY=p, axisRatioZ=q)

    # initial displacement
    r_center = np.array([x0, y0, z0, vx0, vy0, vz0])

    prog_orbit = agama.orbit(ic=r_center, potential=pot_host, time=tend, timestart=0, trajsize=N_orbit, verbose=False)[1]

    xy_prog    = (rot_mat @ prog_orbit[:, 0:3].T)[:2]

    return xy_prog



if __name__ == '__main__':
    # Define the parameters
    # Define parameters
    logM, Rs = 12., 15. 
    p, q = 0.9, 0.8

    logm, rs = 8., 1.  

    x0, y0, z0 = -40., 0., 0.  
    vx0, vy0, vz0 = 0., 150., 0. 

    tend = 2. # Gyr

    rot_mat = np.identity(3)
    rot_mat[0, 1] = 2

    # Call the function
    time0 = timer.time()
    xy_stream, xy_track = Agama_stream(logM, Rs, p, q, logm ,rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, N_stars=500, Nbody=False, seed=True)
    time1 = timer.time()

    plt.title('Time taken: {:.2f} s'.format(time1 - time0))
    plt.scatter(xy_stream[0], xy_stream[1], s=1)
    plt.plot(xy_track[0], xy_track[1], c='r')
    plt.show()