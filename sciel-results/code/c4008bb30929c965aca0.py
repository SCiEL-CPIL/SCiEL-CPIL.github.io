"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the largest common radius for given centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    d_boundary = np.min(np.stack([0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])], axis=1))
    if np.min(d_boundary) < 0:
        return 0.0
    # Minimum pairwise distance / 2
    # Use scipy cdist for efficiency
    from scipy.spatial.distance import cdist
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    d_pairs = np.min(d, axis=1)
    return float(min(np.min(d_boundary), np.min(d_pairs) / 2.0))


def _negative_radius(pts):
    return -_compute_radius(pts)


def _make_hexagonal(m):
    """Hexagonal packing grid, return N points."""
    pts = []
    r = 0.5 / m  # approximate spacing
    for row in range(m + 1):
        x_offset = (row % 2) * 0.5
        for col in range(m):
            x = (col + x_offset) / m - 0.5
            y = row / m - 0.5
            pts.append([x, y])
    pts = np.array(pts)
    # Shuffle and take N
    idx = np.random.permutation(len(pts))
    return pts[idx[:N]]


def _make_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _make_random():
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _make_jittered_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    grid += np.random.normal(0, 0.02, size=grid.shape)
    return grid


def _local_optimize(pts, time_limit):
    """Local optimization using L-BFGS-B with smooth surrogate."""
    start = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Use a smooth objective: minimize negative of a soft-min of distances
    def smooth_obj(x):
        if time.time() - start > time_limit:
            return best_r  # signal to stop
        p = x.reshape(-1, 2)
        # Clip to square
        p = np.clip(p, -0.5, 0.5)
        # Compute pairwise distances
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        d_pairs = np.min(d, axis=1)
        # Distance to boundary
        d_boundary = np.min(np.stack([0.5 - np.abs(p[:, 0]), 0.5 - np.abs(p[:, 1])], axis=1))
        # Soft min using log-sum-exp
        beta = 50.0
        all_d = np.concatenate([d_pairs, d_boundary])
        min_d = -np.log(np.sum(np.exp(-beta * all_d)) / len(all_d)) / beta
        return -min_d
    
    # Optimize with L-BFGS-B
    try:
        res = minimize(
            smooth_obj, pts.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if time.time() - start < time_limit:
            r = _compute_radius(res.x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_pts = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return best_pts, best_r


def construct():
    start = time.time()
    np.random.seed(42)
    
    initializers = [
        _make_grid,
        _make_hexagonal,
        _make_random,
        _make_jittered_grid,
    ]
    
    best_pts = None
    best_r = 0.0
    
    # Try each initializer with local optimization
    for init_func in initializers:
        if time.time() - start > DEADLINE - 5:
            break
        pts = init_func()
        remaining = DEADLINE - time.time() - 5
        if remaining <= 0:
            break
        pts, r = _local_optimize(pts, remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Additional random restarts if time permits
    while time.time() - start < DEADLINE - 10:
        pts = _make_random()
        remaining = DEADLINE - time.time() - 10
        if remaining <= 0:
            break
        pts, r = _local_optimize(pts, remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _make_grid()
    
    # Final validation and return
    if time.time() - start < DEADLINE:
        final_r = _compute_radius(best_pts)
        if final_r <= best_r:
            return best_pts
    
    return best_pts
# EVOLVE-BLOCK-END
