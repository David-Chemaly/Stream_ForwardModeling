import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from tqdm import tqdm
import sympy as sp
import scipy
from scipy.spatial.transform import Rotation

import astropy.units as u
from astropy.constants import G
G = G.to(u.pc * u.Msun**-1 * (u.km / u.s)**2)
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

import gala.potential as gp
import gala.dynamics as gd
import gala.integrate as gi
import gala.units as gu

from gala.units import galactic
from gala.potential import NFWPotential
from gala.dynamics import PhaseSpacePosition, MockStream
from gala.integrate import LeapfrogIntegrator

from astropy.cosmology import default_cosmology
cosmo = default_cosmology.get()
rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * G)).to(u.Msun / u.kpc ** 3)

class NFW():

    def __init__(self,M,c,qxy,qxz, R_orientation, R_rotation):
        self.M = M
        self.c = c
        self.qxy = qxy
        self.qxz = qxz
        self.R_orientation = R_orientation
        self.R_rotation = R_rotation

    def radius_flatten(self,x,y,z):
        x, y, z = self.R_rotation @ self.R_orientation @ np.array([x.value,y.value,z.value]) * x.unit
        return np.sqrt((x)**2+(y/self.qxy)**2+(z/self.qxz)**2)
    
    def A_NFW(self):
        return np.log(1+self.c) - self.c/(1+self.c)
    
    def rho0_Rscube(self):
        return self.M/self.A_NFW()
    
    # Convention 200
    def Rvir_fct_M(self):
        return (3*self.M/(4*np.pi*200*rho_c))**(1/3) 
    
    def Rs_fct_RvirAndc(self):
        return self.Rvir_fct_M()/self.c
        
    # Outputs potential in (km/s)^2
    def potential(self,x,y,z):
        r  = self.radius_flatten(x,y,z)

        Rs = self.Rs_fct_RvirAndc()

        return - G/r * self.M/self.A_NFW() * np.log(1 + r/Rs)
    
    # Outputs acceleration in km/s^2
    def acceleration(self,x,y,z):
        r   = self.radius_flatten(x,y,z)
        Rs  = self.Rs_fct_RvirAndc()

        a_r = -G*self.M/(self.A_NFW()*r**2*(r+Rs)) * ( (r+Rs)*np.log(1 + r/Rs) - r)

        a_x = a_r/r * x
        a_y = a_r/r * y/self.qxy
        a_z = a_r/r * z/self.qxz

        return [a_x.value,a_y.value,a_z.value] * a_x.unit
    
    # Outputs second derivative of the potential in 1/s^2
    def second_derivative_potential(self,x,y,z):
        r   = self.radius_flatten(x,y,z)
        Rs  = self.Rs_fct_RvirAndc()

        factor = G*self.M/(self.A_NFW()*r**3*(r+Rs)**2)

        return factor * ( r*(2*Rs+3*r) - 2*(Rs+r)**2*np.log(1+r/Rs) )
            
    def mass_enclosed(self,r):
        return self.M * (np.log(1 + r/self.Rs_fct_RvirAndc()) - r/(r + self.Rs_fct_RvirAndc()))
    
def LeepFrog(a_fct, x_old, y_old, z_old, vx_old, vy_old, vz_old, dt):

    acc_old = a_fct(x_old,y_old,z_old)

    acc_x_old = acc_old[0].to(u.km/u.s**2)
    acc_y_old = acc_old[1].to(u.km/u.s**2)
    acc_z_old = acc_old[2].to(u.km/u.s**2)
    
    vx_half = vx_old + 0.5*dt*acc_x_old
    vy_half = vy_old + 0.5*dt*acc_y_old
    vz_half = vz_old + 0.5*dt*acc_z_old

    x_new = x_old + dt*vx_half
    y_new = y_old + dt*vy_half
    z_new = z_old + dt*vz_half

    acc_new = a_fct(x_new,y_new,z_new)

    acc_x_new = acc_new[0].to(u.km/u.s**2)
    acc_y_new = acc_new[1].to(u.km/u.s**2)
    acc_z_new = acc_new[2].to(u.km/u.s**2)

    vx_new = vx_half + 0.5*dt*acc_x_new
    vy_new = vy_half + 0.5*dt*acc_y_new
    vz_new = vz_half + 0.5*dt*acc_z_new

    return x_new, y_new, z_new, vx_new, vy_new, vz_new

def Rx(theta):
    """Returns the 3D rotation matrix for a rotation around the X-axis by an angle theta (in radians)."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Ry(theta):
    """Returns the 3D rotation matrix for a rotation around the Y-axis by an angle theta (in radians)."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(theta):
    """Returns the 3D rotation matrix for a rotation around the Z-axis by an angle theta (in radians)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def uniform_random_rotation(data, theta):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Arguments:
        data: vector or set of vectors with dimension (3, n), where n is the
            number of vectors

        theta: uniformal random variable in [0, 1)

    Returns:
        Array of shape (3, n) containing the randomly rotated vectors of x,
        about the mean coordinate of x.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation(x1):
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        # x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # Transpose from (3, n) to (n, 3)
    x = data.T

    # There are two random variables in [0, 1) here (naming is same as paper)
    x1 = theta[0] #np.random.rand()
    x2 = 2 * np.pi * theta[1] #np.random.rand()
    x3 = theta[2] #np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation(x1)
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    x_rot = ((x - mean_coord) @ M) + mean_coord @ M
    data_rot = x_rot.T
    return data_rot

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])

class get_mat():

    def __init__(self, a, b, c, aa, bb):

        self.v1 = np.array([0, 0, 1])

        v2 = np.array([a, b, c])
        self.v2 = v2 / np.sum(v2**2)**.5

        v3 = np.cross(self.v1, self.v2)
        self.v3 = v3 / np.sum(v3**2)**.5

        self.aa = aa
        self.bb = bb
    
    def orientation(self):
        
        angle = np.arccos(np.sum(self.v1 * self.v2))

        return Rotation.from_rotvec(angle * self.v3).as_matrix()
    
    def rotation(self):

        z_new = - (self.aa * self.v2[0] + self.bb * self.v2[1]) / self.v2[2]
        v_new = np.array([self.aa, self.bb, z_new])

        v_norm = v_new / np.sum(v_new**2)**.5

        new_angle = np.arccos(np.sum(v_norm*self.v3)) 

        return Rotation.from_rotvec(new_angle * self.v2).as_matrix()
    
def run(Mass, concentraion, qxy, qxz, pos_init, vel_init, t_end, alpha, beta, gama, aa, bb, N_time):

    # Rotate
    rot_mat = get_mat(alpha, beta, gama, aa, bb)
    R_orientation = rot_mat.orientation()
    R_rotation    = rot_mat.rotation()

    # Define Potential
    halo = NFW(Mass, concentraion, qxy, qxz, R_orientation, R_rotation)

    # Define Initial Conditions
    xp, yp, zp    = pos_init[0], pos_init[1], pos_init[2]
    vxp, vyp, vzp = vel_init[0], vel_init[1], vel_init[2]

    # Define Time
    time = np.linspace(0, t_end, N_time) # * u.Gyr
    dt = time[1] - time[0]

    # Initialize Arrays
    all_pos_p = np.zeros([3, len(time)+1]) * u.kpc
    all_pos_p[:,0] = [xp,yp,zp] 

    # Evolve Orbit
    for tndex, t in enumerate(time):
        xp, yp, zp, vxp, vyp, vzp = LeepFrog(halo.acceleration, xp, yp, zp, vxp, vyp, vzp, dt*u.Gyr)
        all_pos_p[:,int(tndex+1)] = [xp,yp,zp]
    
    return all_pos_p

'''
Gala generated orbits and/or streams
'''

def get_Jacobian(a,b,c):
    # Define the symbols for Cartesian and Spherical coordinates
    x, y, z = sp.symbols('x y z')

    # Define the transformations from Cartesian to Spherical coordinates
    r_expr = sp.sqrt(x**2 + y**2 + z**2)
    theta_expr = sp.acos(z / sp.sqrt(x**2 + y**2 + z**2))
    phi_expr = sp.atan2(y, x)

    # Create the Jacobian matrix
    J = sp.Matrix([
        [r_expr.diff(x), r_expr.diff(y), r_expr.diff(z)],
        [theta_expr.diff(x), theta_expr.diff(y), theta_expr.diff(z)],
        [phi_expr.diff(x), phi_expr.diff(y), phi_expr.diff(z)]
    ])

    # Define a specific point (x, y, z)
    point = {x: a, y: b, z: c}  # Example point

    # Substitute the point into the Jacobian matrix to get numeric values
    J_numeric = J.subs(point)

    return np.array(J_numeric, dtype=float)

def get_rt(wp, pot_NFW, mass_plummer):

    rp = np.linalg.norm( wp.xyz )
    angular_velocity = ( np.linalg.norm( wp.angular_momentum() ) / rp**2 ).to(u.Gyr**-1)

    J = get_Jacobian(wp.x.value, wp.y.value, wp.z.value)
    d2pdr2 = (J.T * pot_NFW.hessian( wp )[:,:,0] * J)[0,0]
    rt = ( G * mass_plummer / (angular_velocity**2 - d2pdr2) ).to(u.kpc**3) **(1/3)
    return rt

class get_mat():

    def __init__(self, a, b, c, aa, bb):

        self.v1 = np.array([0, 0, 1])

        v2 = np.array([a, b, c])
        self.v2 = v2 / np.sum(v2**2)**.5

        v3 = np.cross(self.v1, self.v2)
        self.v3 = v3 / np.sum(v3**2)**.5

        self.aa = aa
        self.bb = bb
    
    def orientation(self):
        
        angle = np.arccos(np.sum(self.v1 * self.v2))

        return Rotation.from_rotvec(angle * self.v3).as_matrix()
    
    def rotation(self):

        z_new = - (self.aa * self.v2[0] + self.bb * self.v2[1]) / self.v2[2]
        v_new = np.array([self.aa, self.bb, z_new])

        v_norm = v_new / np.sum(v_new**2)**.5

        new_angle = np.arccos(np.sum(v_norm*self.v3)) 

        return Rotation.from_rotvec(new_angle * self.v2).as_matrix()
    
class FastRandomRotationMatrices():

    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def rot_z(self):
        theta = 2*np.pi*self.x1
        return np.array([[np.cos(theta), np.sin(theta), 0], 
                        [-np.sin(theta), np.cos(theta), 0], 
                        [0, 0, 1]])

    def rot_v(self):
        theta = 2*np.pi*self.x2
        return np.array( [np.cos(theta)*np.sqrt(self.x3), np.sin(theta)*np.sqrt(self.x3), np.sqrt(1-self.x3)] )
    
    def forward(self):
        R = self.rot_z()
        v = self.rot_v()[:, None]

        M = (2*v@v.T - np.eye(3)) @ R

        return M
    
def RodriguezRotation(kx, ky, kz, tx, ty):
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)
    kx /= k_norm
    ky /= k_norm
    kz /= k_norm

    theta = np.arctan2(ty,tx)

    K = np.array([[0, -kz, ky], 
                [kz, 0, -kx], 
                [-ky, kx, 0]])

    return  np.identity(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return Rotation.from_rotvec(angle * v3).as_matrix()

class MyWishart(scipy.stats._multivariate.wishart_gen):
    def __init__(self, a,b,c,d,e,f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization if necessary
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _standard_rvs(self, n, shape, dim, df, random_state):
        '''
        Adapted from scipy
        '''
        # Random normal variates for off-diagonal elements from U(0,1)
        n_tril = dim * (dim-1) // 2
        covariances = np.array([self.a,
                                self.b,
                                self.c]).reshape(shape+(n_tril,))
        
        # Random chi-square variates for diagonal elements
        variances = np.array([self.d,
                              self.e,
                              self.f]).reshape((dim,) + shape[::-1]).T
                              
        
        # Create the A matri(ces) - lower triangular
        A = np.zeros(shape + (dim, dim))

        # Input the covariances
        size_idx = tuple([slice(None, None, None)]*len(shape))
        tril_idx = np.tril_indices(dim, k=-1)
        A[size_idx + tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(dim)
        A[size_idx + diag_idx] = variances

        return A

def run_Gala(mass_halo, r_s, 
             t_end, 
             pos_p, vel_p, 
             a, b, c,
             kx, ky, kz,
             mass_plummer = 1e8 * u.Msun, 
             r_plummer = 1 * u.kpc, 
             N_time = 100,
             N = 0, 
             factor = 1.5):

    # Rotate
    df    = 3
    scale = np.identity(df)

    my_wishart = MyWishart(a,b,c,kx,ky,kz)
    covariance_matrix = my_wishart.rvs(df, scale)
    eigvals, eigvec   = scipy.linalg.eigh(covariance_matrix)

    q1, q2, q3 = eigvals#**0.5
    rot_mat    = eigvec

    # Define Main Halo Potential
    pot_NFW = gp.NFWPotential(mass_halo, r_s, a=q1, b=q2, c=q3, units=galactic, origin=None, R=rot_mat)

    # Define Time
    time = np.linspace(0, t_end.value, N_time) # * u.Gyr
    dt   = (time[1] - time[0]) * u.Gyr

    step   = int(t_end/dt)
    if N !=0:
        step_N = int(step/N)
    else:
        step_N = int(step/(N+1))

    orbit_pos_p = np.zeros((len(time), 3)) * u.kpc
    orbit_vel_p = np.zeros((len(time), 3)) 
    orbit_pos_p[0] = pos_p
    orbit_vel_p[0] = vel_p

    pos_N = np.zeros((N, 3)) * u.kpc
    vel_N = np.zeros((N, 3)) * u.km/u.s

    orbit_pos_N = np.zeros((len(time), N, 3)) * u.kpc
    orbit_vel_N = np.zeros((len(time), N, 3)) * u.km/u.s

    leading_arg  = []
    trailing_arg = []

    counter = 0
    for i in range(len(time)):

        # Progenitor Phase Space Position
        wp = gd.PhaseSpacePosition(pos = pos_p,
                                   vel = vel_p)
        
        if i % step_N == 0 and N != 0:
            j = i//step_N

            no_rot_NFW = gp.NFWPotential(mass_halo, r_s, a=1, b=1, c=q, units=galactic, origin=None, R=None)
            rt     = get_rt(wp, no_rot_NFW, mass_plummer) * factor
            rp     = np.linalg.norm( wp.xyz )
            theta  = np.arccos(wp.z/rp)
            phi    = np.arctan2(wp.y,wp.x)

            if counter%2 == 0:
                xt1, yt1, zt1 = (rp - rt)*np.sin(theta)*np.cos(phi), (rp - rt)*np.sin(theta)*np.sin(phi), (rp - rt)*np.cos(theta)
                leading_arg.append(i)
            else:
                xt1, yt1, zt1 = (rp + rt)*np.sin(theta)*np.cos(phi), (rp + rt)*np.sin(theta)*np.sin(phi), (rp + rt)*np.cos(theta)
                trailing_arg.append(i)

            # New N starting position
            pos_N[j] = np.array([xt1.value, yt1.value, zt1.value]) * u.kpc #  # tidal radius

            # New N starting velocity
            sig = np.sqrt( G*mass_plummer/(6*np.sqrt(rt**2+r_plummer**2)) ).to(u.km/u.s)
            if counter%2 == 0:
                vel_N[j] = vel_p - np.sign(vel_p)*abs(np.random.normal(0, sig.value)) * u.km/u.s # velocity dispersion
            else:
                vel_N[j] = vel_p + np.sign(vel_p)*abs(np.random.normal(0, sig.value)) * u.km/u.s

            counter += 1

        # Save Progenitor new Position and Velocity
        orbit_pos_p[i] = pos_p
        orbit_vel_p[i] = vel_p

        if N != 0:

            # Save N new Position and Velocity
            orbit_pos_N[i] = pos_N
            orbit_vel_N[i] = vel_N

            # All N in Phase Space Position
            wN = gd.PhaseSpacePosition(pos = pos_N[:j+1].T,
                                    vel = vel_N[:j+1].T)

            # Define Plummer Potential
            pot_plummer  = gp.PlummerPotential(mass_plummer, r_plummer, units=galactic, origin=pos_p, R=None)
            pot_combined = pot_NFW + pot_plummer
            orbit_N = gp.Hamiltonian(pot_combined).integrate_orbit(wN, dt=dt, n_steps=1)
            pos_N[:j+1] = orbit_N.xyz[:, -1].T
            vel_N[:j+1] = orbit_N.v_xyz[:, -1].T
        
        # Progenitor new Position and Velocity
        orbit_p = gp.Hamiltonian(pot_NFW).integrate_orbit(wp, dt=dt, n_steps=1)
        pos_p = orbit_p.xyz[:, -1]
        vel_p = orbit_p.v_xyz[:, -1]
    
    return orbit_pos_p, orbit_pos_N, leading_arg, trailing_arg

