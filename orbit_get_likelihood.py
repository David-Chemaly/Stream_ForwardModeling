import numpy as np
from orbit_get_projected import model_orbit_agama, model_orbit_gala
from orbit_get_data import CubicSpline_fit

BAD_VAL      = 1e50
VERY_BAD_VAL = 1e100

def log_likelihood_agama(params, dict_data):
    xy_model = model_orbit_agama(params)
    x_model = xy_model[:,0]
    y_model = xy_model[:,1]
    r_model = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.unwrap( np.arctan2(y_model, x_model) )

    cs = CubicSpline_fit(theta_model, r_model)

    if cs is None or theta_model.ptp() > 2*np.pi:
        logl = -VERY_BAD_VAL

    else:
        r_data = dict_data['r']
        theta_data = dict_data['theta']
        r_sig = dict_data['r_sig']

        r_model = cs(theta_data)

        if np.isnan(r_model).any():
            logl = -VERY_BAD_VAL * np.sum(np.isnan(r_model))

        else:
            delta_r = r_data - r_model
            logl    = -0.5 * np.sum(delta_r**2 / r_sig**2 )

    return logl

def log_likelihood_gala(params, dict_data):
    xy_model = model_orbit_gala(params)
    x_model = xy_model[:,0]
    y_model = xy_model[:,1]
    r_model = np.sqrt(x_model**2 + y_model**2)
    theta_model = np.unwrap( np.arctan2(y_model, x_model) )

    cs = CubicSpline_fit(theta_model, r_model)

    if cs is None or theta_model.ptp() > 2*np.pi:
        logl = -VERY_BAD_VAL

    else:
        r_data = dict_data['r']
        theta_data = dict_data['theta']
        r_sig = dict_data['r_sig']

        r_model = cs(theta_data)

        if np.isnan(r_model).any():
            logl = -VERY_BAD_VAL * np.sum(np.isnan(r_model))

        else:
            delta_r = r_data - r_model
            logl    = -0.5 * np.sum(delta_r**2 / r_sig**2 )

    return logl
