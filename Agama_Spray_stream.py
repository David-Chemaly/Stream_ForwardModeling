import os, numpy, agama, scipy.special, scipy.integrate, matplotlib.pyplot as plt
pyfalcon=None

# working units: 1 Msun, 1 kpc, 1 km/s
agama.setUnits(length=1, velocity=1, mass=1)

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * numpy.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm

def orbitDF(pot_host, ic, time, timestart, trajsize, mass):
    # integrate the orbit of a massive particle in the host galaxy, accounting for dynamical friction
    if mass == 0:
        return agama.orbit(ic=ic, potential=pot_host, time=time, timestart=timestart, trajsize=trajsize)
    times = numpy.linspace(timestart, timestart+time, trajsize)
    traj = scipy.integrate.odeint(
        lambda xv, t: numpy.hstack((xv[3:6], pot_host.force(xv[0:3], t=t))),
        ic, times)
    return times, traj

def Spray_stream(logM, Rs, q1, q2, q3, logm, rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, N_data=100, N_stars=10000, plot=False):
    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, q2/q3, q1/q3)

    # Set host potential
    # pot_host = agama.Potential(type='NFW', mass=10**logM, scaleradius=Rs)
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, axisRatioY=q2/q3, axisRatioZ=q1/q3)

    # Set satellite potential
    pot_sat  = agama.Potential(type='Plummer',  mass=10**logm, scaleradius=rs)
    initmass = pot_sat.totalMass()

    # Sample stars from satellite density distribution
    df_sat = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
    xv, _ = agama.GalaxyModel(pot_sat, df_sat).sample(N_stars)

    # initial displacement
    r_center = numpy.array([x0, y0, z0, vx0, vy0, vz0])
    xv += r_center

    # create a spherical isotropic DF for the satellite and sample it with particles
    df_sat = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
    Nbody = 10000
    xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(Nbody)


    # parameters for the simulation
    tupd = 1**-3 # interval for plotting and updating the satellite mass for the restricted N-body simulation
    tau  = 2**-8 # timestep of the full N-body sim (typically should be smaller than eps/v, where v is characteristic internal velocity)
    eps  = 0.1   # softening length for the full N-body simulation



    # simulate the evolution of the disrupting satellite using two methods:
    # "restricted N-body" (r_ prefix) and "full N-body" (if available, f_ prefix)
    r_mass   = [initmass]
    r_traj   = [r_center]
    r_xv     = xv.copy()
    time     = 0.0   # current simulation time
    times_t  = [time]
    times_u  = [time]

    while time < tend:
        # Method 1: restricted N-body
        # first determine the trajectory of the satellite centre in the host potential
        # (assuming that it moves as a single massive particle)
        time_center, orbit_center = orbitDF(pot_host=pot_host, ic=r_center, time=tupd, timestart=time, trajsize=round(tupd/tau) + 1, mass=r_mass[-1])
        times_u.append(time_center[-1])
        times_t.extend(time_center[1:])
        r_traj.extend(orbit_center[1:])
        r_center = orbit_center[-1]  # current position and velocity of satellite CoM
        # initialize the time-dependent total potential (host + moving sat) on this time interval
        pot_total = agama.Potential(pot_host,
            agama.Potential(potential=pot_sat, center=numpy.column_stack((time_center, orbit_center))))
        # compute the trajectories of all particles moving in the combined potential of the host galaxy and the moving satellite
        r_xv = numpy.vstack(agama.orbit(ic=r_xv, potential=pot_total, time=tupd, timestart=time, trajsize=1)[:,1])
        # update the potential of the satellite (using a spherical monopole approximation)
        pot_sat = agama.Potential(type='multipole', particles=(r_xv[:,0:3] - r_center[0:3], mass), symmetry='s')
        # determine which particles remain bound to the satellite
        r_bound = pot_sat.potential(r_xv[:,0:3] - r_center[0:3]) + 0.5 * numpy.sum((r_xv[:,3:6] - r_center[3:6])**2, axis=1) < 0
        r_mass.append(numpy.sum(mass[r_bound]))

        time += tupd

    return r_xv[:, 0:3]

if __name__ == '__main__':
    # Define parameters
    logM, Rs = 12., 15. 
    q1, q2, q3 = 1, 1, 1

    logm, rs = 8., 1.  

    x0, y0, z0 = -40., 0., 0.  
    vx0, vy0, vz0 = 0., 150., 0. 

    tend = 1. # Gyr

    rot_mat = numpy.identity(3)
    # rot_mat[0, 1] = 2

    # Get stream
    xyz_stream = Spray_stream(logM, Rs, q1, q2, q3, logm ,rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, plot=True)

    plt.scatter(xyz_stream[:, 0], xyz_stream[:, 1], s=1)
    plt.show()
