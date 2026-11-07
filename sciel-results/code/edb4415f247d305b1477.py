import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 135.0  # safety margin under the 150 s wall-clock limit


def _util(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _mu(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / c
    return np.power(c, -gamma)


def _mu_inv(x, gamma):
    x = np.maximum(x, 1e-300)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / x
    return np.power(x, -1.0 / gamma)


def _cheb_interp(x, xs, ys):
    """Interpolate smooth ys(x); xs increasing. Clamped outside [xs[0], xs[-1]]."""
    if x.size == 0:
        return np.array([], dtype=float)
    n = xs.size
    if n == 1:
        return np.full(x.shape, ys[0], dtype=float)
    j = np.arange(n, dtype=float)
    t = 2.0 * j / (n - 1) - 1.0
    c = 2.0 * np.fft.rfft(ys) / n
    c[0] *= 0.5
    if n % 2 == 0:
        c[-1] *= 0.5
    u = np.clip(2.0 * (x - xs[0]) / (xs[-1] - xs[0]) - 1.0, -1.0, 1.0)
    m = n - 1
    out = np.zeros(x.shape, dtype=float)
    b0, b1 = np.zeros(x.shape, dtype=float), np.zeros(x.shape, dtype=float)
    for k in range(m, -1, -1):
        ck = c[k]
        t0 = b0
        b0 = u * b1 - 0.5 * (t0 + ck)
        b1 = t0
    out += 0.5 * (b0 + b1)
    return out


def _solve_euler(mu_target, m, cmax, gamma, R, beta, p1, p2, v1, v2, xg, xglog):
    """Solve mu(c) = beta R [p1 mu(mu_inv(v1(m'-c)+inc1)) + p2 mu(mu_inv(v2(m'-c)+inc2))]."""
    inc1 = p1 * v1
    inc2 = p2 * v2
    mu_lo = _mu(cmax, gamma)
    mu_hi = _mu(1e-12, gamma)

    def F(mu):
        c = _mu_inv(mu, gamma)
        mprime = R * (m - c)
        a1 = mprime + inc1
        a2 = mprime + inc2
        v1a = _cheb_interp(np.log(np.maximum(a1, 1e-12)), xglog, v1)
        v2a = _cheb_interp(np.log(np.maximum(a2, 1e-12)), xglog, v2)
        m1 = _mu(_mu_inv(v1a, gamma), gamma)
        m2 = _mu(_mu_inv(v2a, gamma), gamma)
        return beta * R * (p1 * m1 + p2 * m2) - mu

    def dFdmu(mu):
        c = _mu_inv(mu, gamma)
        mprime = R * (m - c)
        a1 = mprime + inc1
        a2 = mprime + inc2
        v1a = _cheb_interp(np.log(np.maximum(a1, 1e-12)), xglog, v1)
        v2a = _cheb_interp(np.log(np.maximum(a2, 1e-12)), xglog, v2)
        mu1 = _mu(_mu_inv(v1a, gamma), gamma)
        mu2 = _mu(_mu_inv(v2a, gamma), gamma)
        if abs(gamma - 1.0) < 1e-10:
            dvc = -mu
        else:
            dvc = -mu ** (1.0 + 1.0 / gamma)
        dF = -beta * R * (p1 * mu1 * dvc + p2 * mu2 * dvc)
        if abs(gamma - 1.0) < 1e-10:
            dF -= -1.0 / c
        else:
            dF -= gamma * mu ** (1.0 / gamma)
        return dF

    # initial guess: steady state  mu_ss = (beta R)^{-1/gamma}
    mu = (beta * R) ** (-1.0 / gamma)
    if mu < mu_lo:
        mu = mu_lo
    if mu > mu_hi:
        mu = mu_hi

    lo = mu_lo
    hi = mu_hi
    for _ in range(24):
        Fv = F(mu)
        if abs(Fv) < 1e-13:
            break
        if Fv > 0:
            lo = mu
        else:
            hi = mu
        dFv = dFdmu(mu)
        step = Fv / dFv if abs(dFv) > 1e-300 else 0.0
        mu_new = mu - step
        if not (np.isfinite(mu_new) and mu_new > lo and mu_new < hi):
            mu_new = 0.5 * (lo + hi)
        if abs(mu_new - mu) < 1e-14 * max(1.0, mu):
            mu = mu_new
            break
        mu = mu_new
    c = _mu_inv(mu, gamma)
    if c > cmax:
        c = cmax
    if c < 1e-12:
        c = 1e-12
    return c


def _solve_horizon(h, model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    P = model['transition']
    inc1 = income[0]
    inc2 = income[1]
    p1 = P[0, 1]
    p2 = P[0, 0]

    # log-wealth working grid (clamped to the resource grid for output validity)
    xg = np.clip(np.geomspace(1e-3, 20.0, 200), resource_grid[0], resource_grid[-1])
    xglog = np.log(xg)
    m = xg
    cmax = m * 1.0  # c <= m

    # terminal: consume everything
    v0 = _util(np.maximum(m, 1e-12), gamma)

    # steady-state value (constant in m), used as damping anchor
    ss_c = (1.0 - beta * R) * (inc1 + inc2)
    if ss_c < 1e-9:
        ss_c = 1e-9
    ss_v = beta * R * _util(ss_c, gamma) / (1.0 - beta * R)

    v1 = v0.copy()
    v2 = v0.copy()
    for t in range(1, h + 1):
        w = min(1.0, 0.15 + 0.85 * (t / h))
        v1 = w * v1 + (1.0 - w) * ss_v
        v2 = w * v2 + (1.0 - w) * ss_v
        c = np.empty_like(m)
        for i in range(m.size):
            c[i] = _solve_euler(
                _mu(cmax[i], gamma), m[i], cmax[i], gamma, R, beta, p1, p2, v1, v2, xg, xglog
            )
        mprime = R * (m - c)
        v1a = _cheb_interp(np.log(np.maximum(mprime + inc1, 1e-12)), xglog, v1)
        v2a = _cheb_interp(np.log(np.maximum(mprime + inc2, 1e-12)), xglog, v2)
        vnew = _util(c, gamma) + beta * (p1 * v1a + p2 * v2a)
        v2 = v1
        v1 = vnew
        if time.time() > _DEADLINE:
            break
    return c


def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    P = model['transition']
    h = model['horizon']
    inc1 = income[0]
    inc2 = income[1]
    p1 = P[0, 1]
    p2 = P[0, 0]

    xg = np.clip(np.geomspace(1e-3, 20.0, 200), resource_grid[0], resource_grid[-1])
    xglog = np.log(xg)
    m = xg
    cmax = m * 1.0

    v0 = _util(np.maximum(m, 1e-12), gamma)
    ss_c = (1.0 - beta * R) * (inc1 + inc2)
    if ss_c < 1e-9:
        ss_c = 1e-9
    ss_v = beta * R * _util(ss_c, gamma) / (1.0 - beta * R)

    v1 = v0.copy()
    v2 = v0.copy()
    c = np.ones_like(m)  # safe fallback: consume all

    for t in range(1, h + 1):
        w = min(1.0, 0.15 + 0.85 * (t / h))
        v1 = w * v1 + (1.0 - w) * ss_v
        v2 = w * v2 + (1.0 - w) * ss_v
        c = np.empty_like(m)
        for i in range(m.size):
            c[i] = _solve_euler(
                _mu(cmax[i], gamma), m[i], cmax[i], gamma, R, beta, p1, p2, v1, v2, xg, xglog
            )
        mprime = R * (m - c)
        v1a = _cheb_interp(np.log(np.maximum(mprime + inc1, 1e-12)), xglog, v1)
        v2a = _cheb_interp(np.log(np.maximum(mprime + inc2, 1e-12)), xglog, v2)
        vnew = _util(c, gamma) + beta * (p1 * v1a + p2 * v2a)
        v2 = v1
        v1 = vnew
        if time.time() > _DEADLINE:
            break

    # interpolate onto the resource grid (clamped to grid bounds)
    c_grid = _cheb_interp(np.log(resource_grid), xglog, c)
    c_grid = np.clip(c_grid, 1e-12, resource_grid + 1e-10)
    c_grid = np.minimum(c_grid, resource_grid + 1e-10)

    # assemble (horizon, 2 income states, resource index); both states share the same policy
    cons = np.empty((h, 2, resource_grid.size), dtype=float)
    cons[:, 0, :] = c_grid
    cons[:, 1, :] = c_grid

    # terminal constraint: consume everything
    cons[h - 1, 0, :] = resource_grid
    cons[h - 1, 1, :] = resource_grid

    # hard validity guards
    cons = np.where(np.isfinite(cons), cons, resource_grid)
    cons = np.clip(cons, 1e-12, None)
    cons = np.minimum(cons, resource_grid + 1e-10)
    cons[h - 1, :, :] = resource_grid
    return cons
