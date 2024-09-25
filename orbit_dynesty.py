import os
# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

import numpy as np

from orbit_get_likelihood import log_likelihood_agama
from orbit_get_prior import prior_transform_ndim12


def fit_one_parallele(data, ndim, nlive):

    nthreads = os.cpu_count()
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(log_likelihood_agama,
                                        prior_transform_ndim12,
                                        ndim,
                                        logl_args=(data, ),
                                        nlive=nlive,
                                        sample='rslice',
                                        pool=poo,
                                        queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)
    
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }

def fit_one(data, ndim, nlive):

    dns = dynesty.DynamicNestedSampler(log_likelihood_agama,
                                    prior_transform_ndim12,
                                    ndim,
                                    logl_args=(data, ),
                                    nlive=nlive,
                                    sample='rslice')
    dns.run_nested(n_effective=10000)
    
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }