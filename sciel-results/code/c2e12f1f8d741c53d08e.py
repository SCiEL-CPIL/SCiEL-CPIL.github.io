"""Improved sphere packing: smooth soft-min objective + L-BFGS-B + BCC init."""
N = 100
import time
import numpy as np
from scipy.optimize import minimize

START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # return at least this many seconds before the limit


def _time_left():
    return (START_TIME + DEADLINE - SAFETY_MARGIN) - time.time()


def _radius(pts):
    """Largest common radius for the given centres (validity-checked)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape != (N, 3) or not np.all(np.isfinite(pts)):
        return 0.0
    # inside the cube [-0.5, 0.5]^3
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    if np.min(wall) < -1e-12:
        return 0.0
    # pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    pd = d[iu]
    if pd.min() < -1e-12:
        return 0.0
    r = min(float(np.min(wall)), float(pd.min()) * 0.5)
    return max(r, 0.0)


def _soft_min_objective(x, beta):
    """Smooth surrogate: minimize -softmin(min(wall, pair/2)).
    softmin of a set of values ~= min, with smooth gradients.
    Returns (value, grad) for the objective to MINIMIZE = -softmin."""
    pts = x.reshape(N, 3)
    # wall distances
    wall = 0.5 - np.max(np.abs(pts), axis=1)            # (N,)
    # pairwise half-distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    pd = d[iu] * 0.5                                     # (M,)
    vals = np.concatenate([wall, pd])                    # (N+M,)
    # softmin via log-sum-exp of -vals
    m = np.min(vals)
    z = np.exp(-(vals - m) * beta)
    denom = np.sum(z)
    softmin = m - np.log(denom) / beta
    # d(softmin)/d(vals)
    w = z / denom
    # gradient w.r.t. x
    grad = np.zeros_like(x)
    # wall part: vals[i] = 0.5 - max_k |x_i,k|
    # d(wall_i)/dx = -sign(x) on the argmax coordinate
    absx = np.abs(pts)
    amax = np.argmax(absx, axis=1)
    rows = np.arange(N)
    gwall = np.zeros_like(pts)
    gwall[rows, amax] = -np.sign(pts[rows, amax])
    # pair part: vals = 0.5*|x_i - x_j|, d/dx_i = 0.5*(x_i-x_j)/|..|
    gi, gj = iu
    diff = pts[gi] - pts[gj]
    norm = np.linalg.norm(diff, axis=1)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    gpair = np.zeros_like(pts)
    np.add.at(gpair, gi, 0.5 * diff / norm[:, None])
    np.add.at(gpair, gj, -0.5 * diff / norm[:, None])
    grad += (w[:N] * gwall)
    grad += (w[N:] * gpair)
    # objective = -softmin, so gradient = -grad(softmin)
    return -softmin, -grad


def _build_bcc_init():
    """BCC lattice: corners + body centres, scaled to fill the cube."""
    a = np.cbrt(2.0 / N)  # lattice constant so 2 per cell * (1/a)^3 = N
    n = int(np.ceil(1.0 / a - 1e-9))
    a = 1.0 / n
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = (i + 0.5) * a - 0.5
                # corner of this cell
                pts.append([c - 0.5 * a, c - 0.5 * a, c - 0.5 * a])
                # body centre
                pts.append([c, c, c])
    pts = np.array(pts[:N], dtype=np.float64)
    # pull back to be strictly inside
    pts = np.clip(pts, -0.45, 0.45)
    return pts


def _run_optimize(x0, beta, max_iter, time_budget):
    """Run L-BFGS-B with a time budget; return best pts and radius."""
    best_pts = x0.copy()
    best_r = _radius(x0)

    def fun(x):
        v, g = _soft_min_objective(x, beta)
        return v, g

    x = x0.copy()
    t0 = time.time()
    step = 0
    try:
        while True:
            res = minimize(fun, x, jac=True, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                                    'ftol': 1e-14, 'gtol': 1e-10, 'gtol_rel': 1e-8})
            x = res.x
            r = _radius(x)
            if r > best_r:
                best_r = r
                best_pts = x.copy()
            step += 1
            if time.time() - t0 > time_budget:
                break
            # if converged and no improvement, stop
            if res.status in (0, 1) and r <= best_r:
                break
    except Exception:
        pass
    return best_pts, best_r


def construct():
    global START_TIME
    # If called fresh, reset the clock.
    if _time_left() < 5.0:
        # very little time; return a safe grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    x0 = _build_bcc_init()
    best_pts = x0.copy()
    best_r = _radius(x0)

    # A few restarts with different betas and small random perturbations.
    rng = np.random.default_rng(12345)
    betas = [8.0, 12.0, 20.0, 30.0]
    n_restarts = 4

    for it in range(n_restarts):
        if _time_left() < 15.0:
            break
        beta = betas[it % len(betas)]
        # perturb the starting point slightly (except first)
        if it == 0:
            x0k = x0.copy()
        else:
            x0k = x0 + rng.normal(0, 0.01, size=x0.shape)
            x0k = np.clip(x0k, -0.49, 0.49)
        time_budget = max(5.0, min(30.0, _time_left() * 0.6))
        pts, r = _run_optimize(x0k, beta, max_iter=40, time_budget=time_budget)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Final high-precision polish with a tight beta if time remains.
    if _time_left() > 12.0:
        pts, r = _run_optimize(best_pts, beta=40.0, max_iter=60,
                               time_budget=_time_left() * 0.7)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Ensure validity: clip into the cube (keeps a valid, possibly slightly smaller radius).
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_pts


# Module-level: allow the evaluator to call construct() after import.
# Reset the clock on import so the first construct() call has the full budget.
START_TIME = time.time()
