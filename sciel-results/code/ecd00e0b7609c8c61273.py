"""Packing 200 equal circles in a unit square: L-BFGS-B optimization with pdist."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances using pdist (faster and more memory efficient)
    dists = pdist(pts)
    min_dist = np.min(dists)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _hex_init():
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.9, s_est * 1.1, 10):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s / 2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - s / 2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _objective(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances) to minimize.
    """
    pts = x.reshape(N, 2)

    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()

    # Pairwise distances using pdist
    pairwise_dists = pdist(pts)

    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])

    # Soft min: min(d_i) ≈ -1/alpha * log(sum(exp(-alpha * d_i)))
    # Use log-sum-exp trick for numerical stability
    m = np.min(all_dists)
    log_sum = m + np.log(np.sum(np.exp(-alpha * (all_dists - m))))
    soft_min_val = -(1.0 / alpha) * log_sum

    # We want to maximize min distance, so minimize -min distance
    return -soft_min_val


def _optimize(pts, time_budget, alphas=[20.0, 50.0, 100.0]):
    """
    Run L-BFGS-B optimization on the initial points with increasing alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()

    for alpha in alphas:
        if time.time() - t0 > time_budget:
            break

        def obj(x):
            return _objective(x, alpha)

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'gtol': 1e-5}
            )
            x0 = res.x
        except Exception:
            pass

    return x0.reshape(N, 2)


def construct():
    t_start = time.time()

    initializers = [
        ("hex", _hex_init),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
    ]

    best_pts = None
    best_r = 0.0

    for name, init_fn in initializers:
        if time.time() > DEADLINE - 10:
            break

        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break

        # Allocate time: give hex more time as it's usually better
        if name == "hex":
            time_budget = min(remaining * 0.45, 40.0)
        else:
            time_budget = min(remaining * 0.25, 20.0)

        opt_pts = _optimize(pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)

        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # Final polish: run a short high-alpha optimization on the best result
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        time_budget = min(remaining * 0.4, 10.0)
        opt_pts = _optimize(best_pts, time_budget, alphas=[100.0])
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _hex_init()

    return best_pts
