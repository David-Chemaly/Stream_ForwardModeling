import os, numpy, agama, scipy.special, scipy.integrate, pyfalcon, matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# Define a function to compute the density normalization
def compute_densitynorm(M, Rs, p, q):
    # Simplified example of computing C based on the scale radius and axis ratios
    # In practice, this can be more complex and may involve integrals over the profile
    C = (4 * numpy.pi * Rs**3) / (p * q)
    densitynorm = M / C
    return densitynorm


def Nbody_stream(logM, Rs, q1, q2, q3, logm, rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, N_data=100, N_stars=10000, plot=False):
    # working units: 1 Msun, 1 kpc, 1 km/s
    agama.setUnits(length=1, velocity=1, mass=1)

    # Compute densitynorm
    densitynorm = compute_densitynorm(10**logM, Rs, q2/q3, q1/q3)

    # Set host potential
    # pot_host = agama.Potential(type='NFW', mass=10**logM, scaleradius=Rs)
    pot_host = agama.Potential(type='Spheroid', densitynorm=densitynorm, scaleradius=Rs, gamma=1, alpha=1, beta=3, axisRatioY=q2/q3, axisRatioZ=q1/q3)

    # Set satellite potential
    pot_sat  = agama.Potential(type='Plummer',  mass=10**logm, scaleradius=rs)

    # Sample stars from satellite density distribution
    df_sat = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
    xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(N_stars)

    # initial displacement
    r_center = numpy.array([x0, y0, z0, vx0, vy0, vz0])
    xv += r_center

    # parameters for the simulation
    tupd = 1**-3 # interval for plotting and updating the satellite mass for the restricted N-body simulation
    tau  = 2**-8 # timestep of the full N-body sim (typically should be smaller than eps/v, where v is characteristic internal velocity)
    eps  = 0.1   # softening length for the full N-body simulation

    # simulate the evolution of the disrupting satellite using pyfalcon:
    time     = 0.0   # current simulation time
    f_xv     = xv.copy()
    f_gamma  = (f_xv[:, 0:3]**2).sum(axis=1) - (numpy.array(r_center.copy()[0:3])**2).sum()

    while time < tend:
        if time==0:   # initialize accelerations and potential
            f_acc, _ = pyfalcon.gravity(f_xv[:,0:3], agama.G * mass, eps)
            # f_acc += pot_host.force( (rot_mat @ (f_xv[:,0:3]/numpy.array([1, q2/q3, q1/q3])**2).T).T)
            f_acc += pot_host.force( f_xv[:,0:3] )

        # advance the N-body sim in smaller steps
        f_time = 0
        while f_time < tupd:
            # kick-drift-kick leapfrog method:
            # kick for half-step, using accelerations computed at the end of the previous step
            f_xv[:,3:6] += f_acc * (tau/2)
            # drift for full step
            f_xv[:,0:3] += f_xv[:,3:6] * tau
            # recompute accelerations from self-gravity of the satellite
            # NB: falcON works with natural N-body units in which G=1, so we multiply particle mass passed to falcon by G
            f_acc, _ = pyfalcon.gravity(f_xv[:,0:3], agama.G * mass, eps)
            # add accelerations from the host galaxy
            f_acc += pot_host.force( f_xv[:,0:3] )
            # kick again for half-step
            f_xv[:,3:6] += f_acc * (tau/2)

            f_gamma += (f_xv[:, 0:3]**2).sum(axis=1) - (numpy.array(r_center.copy()[0:3])**2).sum()
            f_time += tau

        time += tupd


    lower_bound = numpy.percentile(f_gamma, 5)
    upper_bound = numpy.percentile(f_gamma, 95)

    arg_keep = numpy.where( (f_gamma >= lower_bound) & (f_gamma <= upper_bound))[0]
    gamma = f_gamma[arg_keep]

    (_, orbit), _ = agama.orbit(potential=pot_host, ic=r_center, time=tend, trajsize=int(tend*1000), dtype=float, der=True)

    xy_orbit  = (rot_mat @ numpy.array(orbit)[:, 0:3].T)[:2]
    xy_stream = (rot_mat @  f_xv[arg_keep, 0:3].T)[:2]
    xy_track  = numpy.zeros((2, N_data))

    gamma_fit = numpy.linspace(gamma.min(), gamma.max(), N_data)

    x_coef = numpy.polyfit(gamma, xy_stream[0], deg=6)
    x_poly = numpy.poly1d(x_coef)
    xy_track[0]  = x_poly(gamma_fit)

    y_coef = numpy.polyfit(gamma, xy_stream[1], deg=6)
    y_poly = numpy.poly1d(y_coef)
    xy_track[1]  = y_poly(numpy.linspace(gamma.min(), gamma.max(), N_data))

    if plot == True:
        plt.scatter(xy_stream[0], xy_stream[1], c=gamma, s=10, label='Stream')
        plt.plot(xy_orbit[0], xy_orbit[1], color='lime', label='Orbit')
        plt.plot(xy_track[0], xy_track[1], color='red', label='Track')
        plt.legend(loc='best', fontsize=10)
        plt.xlabel('X [kpc]', fontsize=16)
        plt.ylabel('Y [kpc]', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    return xy_orbit, xy_stream, xy_track

if __name__ == '__main__':
    # Define parameters
    logM, Rs = 12., 15. 
    q1, q2, q3 = 1, 1, 1

    logm, rs = 8., 1.  

    x0, y0, z0 = -40., 0., 0.  
    vx0, vy0, vz0 = 0., 150., 0. 

    tend = 2. # Gyr

    rot_mat = numpy.identity(3)
    # rot_mat[0, 1] = 2

    # Get stream
    orbit, stream, track = Nbody_stream(logM, Rs, q1, q2, q3, logm ,rs, x0, y0, z0, vx0, vy0, vz0, tend, rot_mat, plot=True)
