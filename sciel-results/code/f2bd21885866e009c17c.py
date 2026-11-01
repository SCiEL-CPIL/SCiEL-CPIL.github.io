"""Packing of 50 equal circles in a unit square via direct maximin optimization."""
import numpy as np
import time

N = 50
HALF = 0.5

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _softmin_score(pts, t):
    """Smooth objective to maximize: soft-min of all constraints (pairwise/2 and boundary).
    We maximize this; larger t = sharper min."""
    # boundary constraints (want >= 0, i.e. distance to edge)
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    # soft min = -logsumexp(-vals)/t
    m = np.min(vals)
    s = np.sum(np.exp(-t * (vals - m)))
    return m - np.log(s) / t


def _violation_score(pts, r_target, t):
    """Smooth objective: minimize total violation of radius r_target.
    Returns a smooth measure of how much the configuration violates the
    constraint that all pairwise distances >= 2*r and boundary distances >= r.
    Lower is better (0 means feasible)."""
    # boundary constraints: want d_i >= r_target
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    # pairwise: want dist_ij/2 >= r_target
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    
    # violations: how much each constraint is violated (positive = violation)
    boundary_viol = np.maximum(0, r_target - d)
    pair_viol = np.maximum(0, r_target - pair)
    
    # Smooth sum of violations using soft-max (log-sum-exp) for differentiability
    # We want to minimize sum of violations; use smooth approximation
    # For numerical stability, use logsumexp on the violations
    all_viol = np.concatenate([boundary_viol, pair_viol])
    # If all zero, return 0
    if np.max(all_viol) < 1e-15:
        return 0.0
    # Smooth max-like penalty: sum of exp(t * viol) - 1, scaled
    # This is differentiable and large when violations exist
    m = np.max(all_viol)
    s = np.sum(np.exp(t * (all_viol - m)))
    return np.log(s) / t + m - 1.0 / t


def _optimize_from(pts0, t, maxiter, deadline):
    """Run L-BFGS-B to maximize the soft-min score. Returns (pts, score)."""
    from scipy.optimize import minimize
    x0 = pts0.ravel()

    def neg_obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_softmin_score(p, t)

    def neg_grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # numerical gradient (cheap enough for N=50, 100 vars)
        eps = 1e-5
        g = np.zeros_like(x)
        base = neg_obj(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            g[i] = (neg_obj(xp) - base) / eps
        return g

    res = minimize(neg_obj, x0, method='L-BFGS-B',
                   jac=neg_grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    pts = res.x.reshape(N, 2)
    # clip to be safe (shouldn't be needed with bounds)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(-res.fun)


def _optimize_violation(pts0, r_target, t, maxiter, deadline):
    """Run L-BFGS-B to minimize violation of radius r_target. Returns (pts, violation)."""
    from scipy.optimize import minimize
    x0 = pts0.ravel()

    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_score(p, r_target, t)

    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        eps = 1e-5
        g = np.zeros_like(x)
        base = obj(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            g[i] = (obj(xp) - base) / eps
        return g

    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(res.fun)


def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _hex_init(rng, jitter=0.0):
    # Hexagonal close-packed lattice, jittered
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        col = 0
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
            col += 1
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _grid_init(rng, jitter=0.0):
    m = 7  # 7x7 = 49, add one
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # add a 50th near center
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    rng = np.random.default_rng(12345)

    best_pts = None
    best_r = 0.0

    def try_optimize(pts0, t, maxiter):
        nonlocal best_pts, best_r
        if time.time() > deadline:
            return
        try:
            pts, score = _optimize_from(pts0, t, maxiter, deadline)
            r = _compute_r(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            pass

    def try_violation_optimize(pts0, r_target, t, maxiter):
        nonlocal best_pts, best_r
        if time.time() > deadline:
            return
        try:
            pts, viol = _optimize_violation(pts0, r_target, t, maxiter, deadline)
            r = _compute_r(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            pass

    # Phase 1: multi-start with moderate sharpness to get a good initial solution
    inits = []
    # hexagonal
    inits.append(_hex_init(rng, jitter=0.01))
    inits.append(_hex_init(rng, jitter=0.03))
    inits.append(_grid_init(rng, jitter=0.02))
    inits.append(_grid_init(rng, jitter=0.05))
    # random
    for _ in range(4):
        inits.append(_random_init(rng))

    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        t0 = 20.0
        maxiter = 150
        try_optimize(init, t0, maxiter)

    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Phase 2: Binary search on radius using violation minimization
    # Start from current best_r, try to push higher
    r_low = best_r
    r_high = min(r_low + 0.01, 0.1)  # upper bound estimate
    
    # Binary search: find max feasible radius
    for iteration in range(15):
        if time.time() > deadline - 15.0:
            break
        r_mid = (r_low + r_high) / 2.0
        
        # Try to achieve r_mid from current best configuration
        t_val = 50.0
        maxiter = 200
        try_violation_optimize(best_pts, r_mid, t_val, maxiter)
        
        # Check if we achieved at least r_mid
        r_achieved = _compute_r(best_pts)
        if r_achieved >= r_mid - 1e-8:
            r_low = r_mid
            # Update best if improved
            if r_achieved > best_r:
                best_r = r_achieved
        else:
            r_high = r_mid
        
        # If we can't even achieve r_low, shrink
        if r_low < 0.01:
            break

    # Phase 3: Polish from the best found with sharp soft-min
    if time.time() < deadline - 10.0:
        for t in [80.0, 150.0, 300.0]:
            if time.time() > deadline - 5.0:
                break
            try_optimize(best_pts, t, 200)

    # Phase 4: Final binary search polish with tighter tolerance
    if time.time() < deadline - 10.0:
        r_low = best_r
        r_high = best_r + 0.005
        for iteration in range(10):
            if time.time() > deadline - 5.0:
                break
            r_mid = (r_low + r_high) / 2.0
            t_val = 100.0
            maxiter = 300
            try_violation_optimize(best_pts, r_mid, t_val, maxiter)
            r_achieved = _compute_r(best_pts)
            if r_achieved >= r_mid - 1e-9:
                r_low = r_mid
                if r_achieved > best_r:
                    best_r = r_achieved
            else:
                r_high = r_mid

    return best_pts
