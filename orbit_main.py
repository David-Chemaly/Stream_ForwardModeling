import os
import pickle
import multiprocessing
import numpy as np

from orbit_get_data import get_data_prior
from orbit_dynesty import fit_one
from orbit_get_likelihood import log_likelihood


def run_in_parallel(q_true, seed, ndim, nlive, dir_save, sigma, N):
    # Create directory if it doesn't exist
    os.makedirs(dir_save, exist_ok=True)

    # Create a list of seeds, one for each process
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, int(1e5), N)

    # Prepare arguments for each process
    args = [(id, s, q_true, ndim, nlive, sigma, dir_save) for id, s in enumerate(seeds)]

    # Use multiprocessing Pool to run the function in parallel
    print(f'Running {N} processes in parallel with {os.cpu_count()} cores')
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(main, args)

def main(id, seed, q_true, ndim, nlive, sigma, dir_save):

    dict_data, params_data = get_data_prior(q_true, ndim, seed=seed, sigma=sigma)
    print(log_likelihood(params_data, dict_data))

    # Save dict_result as a pickle file
    save_stream = f'{dir_save}/xx_{id+1:03d}'
    os.makedirs(save_stream, exist_ok=True)

    with open(f'{save_stream}/dict_data.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    np.savetxt(f'{save_stream}/params.txt', params_data)

    dict_result = fit_one(dict_data, ndim, nlive)

    with open(f'{save_stream}/dict_result.pkl', 'wb') as f:
        pickle.dump(dict_result, f)