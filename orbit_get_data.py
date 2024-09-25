import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import agama
# working units: 1 Msun, 1 kpc, 1 km/s
agama.setUnits(length=1, velocity=1, mass=1)


def get_data_model(xy_stream, sigma=1, n_ang=18):
    # Get the data
    x = xy_stream[:, 0]
    y = xy_stream[:, 1]

    # Get the radius
    r = (x**2 + y**2)**.5

    # Get the angle
    theta = np.unwrap( np.arctan2(y, x) )

    cs = CubicSpline_fit(theta, r)

    theta_bin = np.arange(0, 360, 360/n_ang) * np.pi/180

    if cs is None:
        return None
    
    else:
        r_bin = cs(theta_bin)

        arg_in = ~np.isnan(r_bin)

        if n_ang//4 < arg_in.sum() < n_ang:

            theta_data = theta_bin[arg_in]
            r_data     = r_bin[arg_in]
            x_data = r_data * np.cos(theta_data)
            y_data = r_data * np.sin(theta_data)

            if sigma == 0:
                noise = 0
            else:
                noise = np.random.normal(0, sigma, len(r_data))

            dict_data = {'theta': theta_data, 'r': r_data+noise, 'x': x_data, 'y': y_data, 'r_sig': sigma}

            return dict_data
        
        else:
            return None

def CubicSpline_fit(theta, r, n_ang=18):
    if (np.diff(theta) <= 0).any():
        # print('Theta is not stricly increasing')
        return None
    else:
        # Fit the cubic spline
        cs = CubicSpline(theta, r, extrapolate=False)

        return cs
    
def get_data_prior(fct_prior, fct_model, q_true, ndim, seed=42, sigma=1, n_ang=18):
    rng = np.random.RandomState(seed)

    correct = False
    while not correct:
        p = rng.uniform(size=ndim)
        params = np.array( fct_prior(p) )
        params[2] = q_true

        xy_stream = fct_model(params)

        dict_data = get_data_model(xy_stream, sigma=sigma, n_ang=n_ang)

        if dict_data is not None:
            correct = True

    return dict_data, params

