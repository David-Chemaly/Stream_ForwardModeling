# Third-party
import astropy.units as auni
import astropy.coordinates as coord
import numpy as np
from scipy.spatial.transform import Rotation
# Gala
import gala.dynamics as gd
import gala.potential as gp
import scipy.special
import dynesty
import multiprocessing as mp
# from idlsave import idlsave

coord.galactocentric_frame_defaults.set('v4.0')

BAD_VAL = -1e100


def get_mat(x, y, z):
    v1 = np.array([0, 0, 1])
    v2 = np.array([x, y, z])
    v2 = v2 / np.sum(v2**2)**.5
    angle = np.arccos(np.sum(v1 * v2))
    v3 = np.cross(v1, v2)
    v3 = v3 / np.sum(v3**2)**.5
    return Rotation.from_rotvec(angle * v3).as_matrix()


def cube(p):
    x, y, z, vx, vy, vz, dirx, diry, dirz, lmass, lrs, q, t_total = p
    x1, y1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [.5 + .5 * x, .49 + 0.01 * y, z]
    ]
    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx, vy / 2. + .5, vz]
    ]
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz / 2]
    ]
    lmass1 = (11.5 + lmass)
    lrs1 = (.5 + lrs)
    q1 = .6 + .4 * q
    t_total1 = 10**(2 + t_total)
    return [
        x1, y1, z1, vx1, vy1, vz1, dirx1, diry1, dirz1, lmass1, lrs1, q1,
        t_total1
    ]


def get_ang(x, y):
    ang = np.arctan2(y, x)
    R = (x**2 + y**2)**.5
    ang = np.unwrap(ang)
    sign = np.diff(ang) > 0
    if (not np.all(sign) and
            not np.all(~sign)) or np.abs(ang.max() - ang.min()) > 2 * np.pi:
        return None
    else:
        return ang, np.log10(R)


def chisq_comp(d_ang, logd, elogd, m_ang, II):
    min_m_ang = m_ang.min()
    max_m_ang = m_ang.max()
    d_ang = (d_ang + 10 * np.pi - min_m_ang) % (2 * np.pi) + min_m_ang
    # min_model min_data max_data max_model
    # how it should be
    if ((d_ang > min_m_ang) & (d_ang < max_m_ang)).all():
        penalty = max(d_ang.min() - min_m_ang - 0.1, 0)**2 + max(
            max_m_ang - d_ang.max() - 0.1, 0)**2
        logl = -.5 * np.sum(((II(d_ang) - logd) / elogd)**2) - 10 * penalty
    else:
        logl = BAD_VAL
        penalty = max((np.maximum(d_ang, min_m_ang) - min_m_ang)**2)
        penalty += max((np.minimum(d_ang, max_m_ang) - max_m_ang)**2)
        logl = logl - np.abs(BAD_VAL) / 10000 * penalty
        # print('oops2', penalty)
    return logl


def get_interpol(ang, log10R):
    if ang[1] < ang[0]:
        ang = ang[::-1]

    II = scipy.interpolate.CubicSpline(ang, log10R)
    return II


def like(p, data=None, getData=False, getModel=False, rng=None):
    x, y, z, vx, vy, vz, dirx, diry, dirz, mass, rs, q, t_total = p
    mass, rs = 10**mass, 10**rs
    n_steps = 1000
    units = [auni.kpc, auni.km / auni.s, auni.Msun, auni.Gyr, auni.rad]
    w0 = gd.PhaseSpacePosition(
        pos=np.array([x, y, z]) * auni.kpc,
        vel=np.array([vx, vy, vz]) * auni.km / auni.s,
    )

    mat = get_mat(dirx, diry, dirz)

    pot = gp.NFWPotential(mass, rs, 1, 1, q, R=mat, units=units)
    # t_total = 200
    orbit = pot.integrate_orbit(w0,
                                dt=t_total / n_steps * auni.Myr,
                                n_steps=n_steps)
    xout, yout, _ = orbit.x.to_value(auni.kpc), orbit.y.to_value(
        auni.kpc), orbit.z.to_value(auni.kpc)
    angRes = get_ang(xout, yout)
    if angRes is None:
        if getData or getModel:
            return None
        # print('oops1')
        return 2 * BAD_VAL
    m_ang, m_logr = angRes
    m_II = get_interpol(m_ang, m_logr)
    if getData:
        n_pt = 20
        d_log10r_err = rng.uniform(0.005, 0.01, size=n_pt)
        ang_grid = np.linspace(m_ang[0], m_ang[-1], n_pt)
        d_log10r = m_II(ang_grid) + d_log10r_err * rng.normal(size=n_pt)
        return ang_grid, d_log10r, d_log10r_err
    if getModel:
        return orbit, xout, yout

    d_ang, d_log10r, d_log10r_err = data
    logl = chisq_comp(d_ang, d_log10r, d_log10r_err, m_ang, m_II)
    return logl


def getdatas(n_dat, seed=None):
    q = 0.9
    lmass = 12
    lrs = 1
    time = 200
    rng = np.random.default_rng(seed)
    datas = []
    trueps = []
    while len(datas) < n_dat:
        xs = rng.normal(size=3) * 40
        xs[1] = 0
        xs[0] = np.abs(xs[0])
        vs = rng.normal(size=3) * 70
        vs[1] = np.abs(vs[1])
        vec = rng.normal(size=3)
        vec[-1] = np.abs(vec[-1])
        time = rng.uniform(200, 800)
        #    x, y, z, vx, vy, vz, dirx, diry, dirz, lmas, lrs, q, t_total = p
        pvec = np.concatenate([xs, vs, vec, [lmass, lrs, q, time]])
        curd = like(pvec, getData=True, rng=rng)
        if curd is not None and curd[0].ptp() > 1.5 and curd[1].min(
        ) > 0.5 and curd[1].ptp() < 2:
            datas.append(curd)
            trueps.append(pvec)
            xpvec = pvec * 1
            xpvec[1] = -1e-5
            xpvec[-1] *= 1.001
            print(like(xpvec, data=(curd)))
    return datas, trueps


def testfit(seed=None):
    dat = getdatas(1, seed=seed)[0][0]
    ndim = 13
    nthreads = 36
    # poo = None
    rstate = np.random.default_rng(seed)
    with mp.Pool(nthreads) as poo:
        # if True:
        dns = dynesty.DynamicNestedSampler(
            like,
            cube,
            ndim,
            logl_args=(dat, ),
            nlive=1200,  #4000,
            rstate=rstate,
            sample='rslice',  # 'rwalk',  # slice',
            # slices=100,
            # walks=100,
            pool=poo,
            queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)
    return dns

    # poo = None


def proc_one(seed):
    dats, trueps = getdatas(1, seed=seed)
    dats, trueps = dats[0], trueps[0]
    fitres = fit_one(dats, seed)
    del fitres['dns']
    idlsave.save('saves/xx_%04d.psav' % seed, 'fitres,curdat,truep', fitres,
                 dats, trueps)


def fit_one(dat, seed=None):
    nlive = 1200
    n_eff = 10000
    sampler, kw = 'rwalk', {'walks': None}
    sampler, kw = 'rslice', {}
    ndim = 13

    rstate = np.random.default_rng(seed)
    dns = dynesty.DynamicNestedSampler(like,
                                       cube,
                                       ndim,
                                       logl_args=(dat, ),
                                       nlive=nlive,
                                       rstate=rstate,
                                       sample=sampler,
                                       **kw)
    dns.run_nested(n_effective=n_eff)
    import dynesty.utils as dyut
    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl = res.logl[inds]
    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }


def domany(ndat):
    dats, trueps = getdatas(ndat, seed=44)
    idlsave.save('xdat.psav', 'dats,trueps', dats, trueps)
    nthreads = 36
    # poo = None
    with mp.Pool(nthreads) as poo:
        # if True:
        res = []
        for i, curd in enumerate(dats):
            res.append(poo.apply_async(fit_one, (curd, i)))
        for i in range(len(dats)):
            curr = res[i].get()
            idlsave.save('saves/xx_%03d.psav' % i, 'curr,curd', curr, dats[i])