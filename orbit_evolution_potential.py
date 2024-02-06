import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from astropy.constants import G
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