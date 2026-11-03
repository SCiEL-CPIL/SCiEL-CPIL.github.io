"""Repulsion-dynamics + smooth soft-min polish for packing 200 equal circles."""
import time
import numpy as np

N = 200
_LIMIT = 150.0
_START = time.time()
_DEADLINE = _START + _LIMIT - 6.0   # safety margin before the 150 s kill


def _radius(pts):
    """Raw score: largest common radius for centres pts (side-1 square at origin)."""
    d = np.abs(pts[:, None, :] - pts[None, :, :])
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() * 0.5
    min_wall = (0.5 - np.abs(pts)).min()
    return float(min(min_pair, min_wall))


def _softmin_logsumexp(d, t):
    """Soft-min of a 2D distance array via log-sum-exp (avoids overflow)."""
    m = d.min(axis=(0, 1), keepdims=True)
    e = np.exp(t * (d - m))
    s = e.sum(axis=(0, 1), keepdims=True)
    return m - np.log(s) / t


def _soft_obj(pts, t):
    """Smooth surrogate of -radius: penalises small inter-circle distances and
    wall encroachment via log-sum-exp. Returns (value, grad)."""
    p = pts
    d = np.abs(p[:, None, :] - p[None, :, :])
    np.fill_diagonal(d, np.inf)
    dp = d - p.min(axis=(0, 1), keepdims=True)
    ep = np.exp(t * dp)
    sp = ep.sum(axis=(0, 1), keepdims=True)
    lp = p.min(axis=(0, 1), keepdims=True) - np.log(sp) / t
    # wall distances
    aw = np.abs(p)
    dw = 0.5 - aw
    dwh = dw - dw.min(axis=(0, 1), keepdims=True)
    ew = np.exp(t * dwh)
    sw = ew.sum(axis=(0, 1), keepdims=True)
    lw = dw.min(axis=(0, 1), keepdims=True) - np.log(sw) / t
    val = 0.5 * (lp + lw)
    # gradients
    gp = -(ep / sp) / t
    np.fill_diagonal(gp, 0.0)
    gw = -(ew / sw) / t
    g = 0.5 * (gp.sum(axis=1) + gw)
    # sign of |.|
    g *= np.sign(p)
    return -val, g


def _repulsion(pts, steps=400, k_wall=2.0, dt0=1e-5, gamma=0.9):
    """Rigid-body repulsion: inverse-square inter-circle + quadratic wall repulsion."""
    pts = pts.copy()
    dt = dt0
    for _ in range(steps):
        d = np.abs(pts[:, None, :] - pts[None, :, :])
        np.fill_diagonal(d, np.inf)
        d2 = d * d
        np.fill_diagonal(d2, np.inf)
        f = d * (1.0 / (d2 * d2))
        fsum = f.sum(axis=1)
        # wall repulsion (quadratic)
        aw = np.abs(pts)
        wall = np.zeros_like(pts)
        for k in range(2):
            s = np.sign(pts[:, k])
            wall[:, k] = k_wall * (0.5 - aw[:, k]) * s
        vel = pts * fsum + wall
        v = np.linalg.norm(vel, axis=1, keepdims=True)
        v[v == 0] = 1.0
        vel = vel / v
        pts += dt * vel
        dt *= gamma
    return pts


def _polish(pts, t, time_budget):
    from scipy.optimize import minimize
    b = np.array([-0.5, 0.5]) * np.ones(N)
    best = pts.copy()
    best_r = _radius(pts)
    t_end = time.time() + time_budget
    while time.time() < t_end:
        try:
            res = minimize(_soft_obj, best, jac=True, method='L-BFGS-B',
                           bounds=list(zip(b, b)),
                           options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-9,
                                    'maxls': 30, 'maxfun': 6000})
        except Exception:
            break
        if res.x is not None and np.all(np.isfinite(res.x)):
            r = _radius(res.x)
            if r > best_r:
                best_r = r
                best = res.x.copy()
        if time.time() >= t_end:
            break
    return best, best_r


def _grid_init(m, seed):
    rng = np.random.default_rng(seed)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    base = np.array([(x, y) for y in ticks for x in ticks])[:N]
    jitter = (rng.random((N, 2)) - 0.5) * 0.5 / m
    return base + jitter


def construct():
    # 1) grid + repulsion + polish
    m = int(np.ceil(np.sqrt(N)))
    pts = _grid_init(m, 0)
    pts = _repulsion(pts, steps=500)
    pts, r = _polish(pts, t=30.0, time_budget=40.0)
    best_pts, best_r = pts, r
    if time.time() > _DEADLINE:
        return best_pts

    # 2) random multi-start (repulsion + polish), tighten soft-min
    rng = np.random.default_rng(12345)
    t_sched = [60.0, 120.0, 240.0]
    for i, t in enumerate(t_sched):
        if time.time() > _DEADLINE:
            break
        init = rng.uniform(-0.45, 0.45, size=(N, 2))
        p = _repulsion(init, steps=300)
        p, r = _polish(p, t=t, time_budget=28.0)
        if r > best_r:
            best_r, best_pts = r, p

    # 3) final high-precision polish on the best found
    if time.time() < _DEADLINE - 8.0:
        p, r = _polish(best_pts, t=500.0, time_budget=_DEADLINE - time.time() - 4.0)
        if r > best_r:
            best_r, best_pts = r, p

    # ensure validity (clip tiny drift)
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
