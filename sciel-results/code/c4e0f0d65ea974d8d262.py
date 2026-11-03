"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_DEADLINE = 110.0  # safety margin below the 120s limit
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _triangle_areas_all(points):
    """Return all C(n,3) triangle areas as a flat array (vectorized)."""
    n = points.shape[0]
    # For each triple (i, j, k) with i<j<k, area = 0.5*|cross(pj-pi, pk-pi)|
    # Vectorize over pairs (i,j) and then over k > j
    # Simpler: use broadcasting on index arrays
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas_all(points)
    if areas.size == 0:
        return 0.0
    return float(areas.min() / hull_area)


def _params_to_points(params):
    """
    params: 2N values = [theta_0..theta_{N-1}, r_0..r_{N-1}]
    theta: strictly increasing angles in [0, 2*pi), wrapping handled.
    r: positive radii.
    Points placed on convex hull of a star-shaped polygon.
    """
    n = N
    theta = params[:n]
    r = params[n:]
    # Normalize theta to be strictly increasing and wrap to [0, 2*pi)
    theta = np.sort(theta)
    # Make them strictly increasing by adding tiny epsilon if needed
    # Wrap: subtract mean so they center around 0, then add 2pi to negatives
    theta = theta - theta[0]
    # Ensure they span roughly 2*pi (convex hull of star-shaped)
    # Actually for a convex polygon, angles should be in increasing order covering full circle
    # Let's normalize: subtract first, then if last < 2*pi - small, scale? 
    # Better: treat theta as relative angles, normalize so max - min = 2*pi
    span = theta[-1] - theta[0]
    if span < 1e-12:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    else:
        theta = theta * (2 * np.pi / span)
    # Ensure strictly increasing with small gaps
    for i in range(1, n):
        if theta[i] - theta[i-1] < 1e-9:
            theta[i] = theta[i-1] + 1e-9
    # Re-normalize after fixing
    span = theta[-1] - theta[0]
    if span > 1e-12:
        theta = theta * (2 * np.pi / span)
    r = np.maximum(r, 0.1)  # keep positive
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _neg_objective(params):
    points = _params_to_points(params)
    s = _score(points)
    if s <= 0:
        return 1e6
    return -s


def _nm_single(x0, maxiter=200):
    res = minimize(_neg_objective, x0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
    if res.success or res.fun < 1e5:
        return res.x, -res.fun
    return x0, _score(_params_to_points(x0))


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 110.0

    best_params = None
    best_score = 0.0

    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = []

    # 1. Regular polygon (equal angles, equal radii)
    theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_reg = np.ones(N)
    starts.append(np.concatenate([theta_reg, r_reg]))

    # 2. Regular polygon with varied radii
    for trial in range(8):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        r = rng.uniform(0.7, 1.3, N)
        starts.append(np.concatenate([theta, r]))

    # 3. Random angles and radii
    for trial in range(12):
        theta = rng.uniform(0, 2 * np.pi, N)
        r = rng.uniform(0.5, 1.5, N)
        starts.append(np.concatenate([theta, r]))

    # 4. Clustered angles (some close, some far)
    for trial in range(6):
        theta = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Add some clustering
        for _ in range(3):
            i = rng.integers(0, N-1)
            theta[i+1] = theta[i] + rng.uniform(0.05, 0.3)
        theta = np.sort(theta)
        r = rng.uniform(0.6, 1.4, N)
        starts.append(np.concatenate([theta, r]))

    # Evaluate each start with Nelder-Mead
    for x0 in starts:
        if _time_left() < 5.0:
            break
        params, score = _nm_single(x0, maxiter=150)
        if score > best_score:
            best_score = score
            best_params = params.copy()

    # If we have time, do a second round with perturbed best
    if best_params is not None and _time_left() > 15.0:
        for trial in range(5):
            if _time_left() < 5.0:
                break
            perturbation = rng.normal(0, 0.1, 2 * N)
            x0 = best_params + perturbation
            params, score = _nm_single(x0, maxiter=100)
            if score > best_score:
                best_score = score
                best_params = params.copy()

    if best_params is None:
        # Fallback: regular polygon
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.ones(N)
        best_params = np.concatenate([theta, r])

    points = _params_to_points(best_params)

    # Final sanity check: ensure it's a valid point set
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.ones(N)
        points = _params_to_points(np.concatenate([theta, r]))

    return points
# EVOLVE-BLOCK-END
