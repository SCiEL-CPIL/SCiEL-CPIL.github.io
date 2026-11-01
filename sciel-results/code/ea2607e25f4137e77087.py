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
    deadline = start + 130.0  # safety margin before 150s
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

    # Phase 1: multi-start with moderate sharpness
    inits = []
    # hexagonal
    inits.append(_hex_init(rng, jitter=0.01))
    inits.append(_hex_init(rng, jitter=0.03))
    inits.append(_grid_init(rng, jitter=0.02))
    inits.append(_grid_init(rng, jitter=0.05))
    # random
    for _ in range(6):
        inits.append(_random_init(rng))

    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        # Start with softer t, then sharpen
        t0 = 20.0
        maxiter = 200
        try_optimize(init, t0, maxiter)

    # Phase 2: sharpen from the best found
    if best_pts is not None and time.time() < deadline - 10.0:
        for t in [40.0, 80.0, 150.0]:
            if time.time() > deadline - 5.0:
                break
            try_optimize(best_pts, t, 300)

    # Phase 3: binary-search refinement on radius
    # Try to nudge: check if we can increase r by solving feasibility
    if best_pts is not None and time.time() < deadline - 5.0:
        # Local polish: do a few more sharp iterations
        try_optimize(best_pts, 300.0, 500)

    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
