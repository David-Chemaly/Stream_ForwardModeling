import scipy

def prior_transform_ndim12(p):
    logM, Rs, q, dirx, diry, dirz, \
    x0, z0, vx0, vy0, vz0, \
    t_end = p

    logM1  = (11 + 2*logM)
    Rs1    = (5 + 20*Rs)
    q1     = q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    x1, z1 = [
        scipy.special.ndtri(_) * 100 for _ in [0.5 + x0/2, z0]
    ]
    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 100 for _ in [vx0, 0.5 + vy0/2, vz0]
    ]

    t_end1 = 1.5*t_end

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            x1, z1, vx1, vy1, vz1, 
            t_end1]