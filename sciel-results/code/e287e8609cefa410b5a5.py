"""Rewritten solver: vectorized backward induction, Newton polish on Euler."""
import numpy as np
import time

_CMIN = 1e-12


def _u(c, gamma):
    c = np.asarray(c, dtype=float)
    c = np.clip(c, _CMIN, None)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 15.0

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model.get('R', 1.02))
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    zN = len(income)
    grid = np.asarray(resource_grid, dtype=float)
    n = len(grid)

    md = np.geomspace(0.001, 40, 4001)

    def V_terminal(m):
        if gamma == 1:
            return np.log(np.clip(m, _CMIN, None))
        return np.power(np.clip(m, _CMIN, None), 1.0 - gamma) / (1.0 - gamma)

    Vterm = V_terminal(md)

    grid_defs = [101, 201, 401]
    if budget >= 600:
        grid_defs.append(801)
    nf = grid_defs[-1]

    c_prev = None
    m_g_prev = None
    c_store = None

    for gidx, ng in enumerate(grid_defs):
        if time.time() > DEADLINE - MARGIN:
            break
        m_g = np.geomspace(0.001, 40, ng)
        val = [Vterm.copy() for _ in range(zN)]
        c_t = [None] * horizon
        c_t[horizon - 1] = [m_g.copy() for _ in range(zN)]
        for t in range(horizon - 2, -1, -1):
            if time.time() > DEADLINE - MARGIN:
                break
            if c_prev is not None:
                c_now = [np.interp(m_g, m_g_prev, c_prev[t][z]) for z in range(zN)]
            else:
                c_now = [0.5 * m_g for _ in range(zN)]
            for it in range(14):
                cn = [np.clip(cf, _CMIN, None) for cf in c_now]
                Q = [None] * zN
                Eup = [None] * zN
                for z in range(zN):
                    mc = m_g - cn[z]
                    mnext = R * mc + income[None, :]
                    contv = np.array([np.interp(mnext[zp], md, val[zp]) for zp in range(zN)])
                    Q[z] = (trans[z, None, :] * contv).sum(axis=0)
                    cprime = np.clip(mnext, _CMIN, None)
                    uprime_cprime = np.power(cprime, 1.0 - gamma)
                    Eup[z] = (trans[z, None, :] * uprime_cprime).sum(axis=0)
                c_new = [None] * zN
                maxstep = 0.0
                for z in range(zN):
                    u1 = np.power(np.clip(cn[z], _CMIN, None), 1.0 - gamma)
                    f = u1 - beta * R * Eup[z]
                    fp = -(1.0 - gamma) * u1 / np.clip(cn[z], _CMIN, None)
                    step = f / fp
                    cnv = np.clip(cn[z] + step, _CMIN, m_g - 1e-9)
                    c_new[z] = cnv
                    maxstep = max(maxstep, float(np.max(np.abs(cnv - cn[z]))))
                c_now = c_new
                if maxstep < 1e-9:
                    break
            c_t[t] = c_now
            c_md = [np.interp(md, m_g, cf) for cf in c_now]
            for z in range(zN):
                Q_md = np.interp(md, m_g, Q[z])
                val[z] = _u(c_md[z], gamma) + beta * Q_md
        c_prev = c_t
        m_g_prev = m_g
        if ng == nf:
            c_store = c_t

    c_out = np.zeros((horizon, zN, n), dtype=float)
    for t in range(horizon):
        for z in range(zN):
            c_out[t, z] = np.interp(grid, m_g_prev, c_store[t][z])
    c_out = np.clip(c_out, _CMIN, None)
    c_out = np.minimum(c_out, grid[None, None, :] + 1e-10)
    c_out[horizon - 1] = grid[None, None, :]
    return c_out
