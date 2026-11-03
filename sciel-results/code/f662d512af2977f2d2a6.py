"""Rewritten program for 50 equal circles in a unit circle."""
N = 50
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _score(pts):
    """Compute the packing radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Boundary constraint
    dist_center = np.linalg.norm(pts, axis=1)
    r_boundary = 1.0 - dist_center
    if np.any(r_boundary <= 0):
        return 0.0
    r_min = np.min(r_boundary)
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask out self-distances
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_min, r_pair)


def _negative_score(pts_flat):
    """Objective function for optimization (minimize negative score)."""
    pts = pts_flat.reshape(-1, 2)
    return -_score(pts)


def _init_hexagonal():
    """Initialize points on a hexagonal lattice clipped to the unit disk."""
    # Start with a dense hexagonal grid
    spacing = 0.25
    points = []
    k = 8
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.8, N)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Select N points closest to center for a compact initial configuration
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    return points


def _init_random():
    """Initialize with random points inside the unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin

    # Try multiple initializations and optimization methods
    best_pts = None
    best_score = 0.0

    # Strategy 1: Hexagonal lattice initialization with L-BFGS
    for restart in range(5):
        if time.time() > deadline:
            break
        if restart == 0:
            pts = _init_hexagonal()
        else:
            # Perturb the hexagonal init
            pts = _init_hexagonal()
            noise = np.random.normal(0, 0.05, pts.shape)
            pts = pts + noise
            # Project back inside the unit circle
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 0.9
            pts[mask] *= 0.9 / dists[mask, np.newaxis]

        res = minimize(_negative_score, pts.flatten(), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < best_score:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_pts = res.x.reshape(-1, 2)

    # Strategy 2: Random initialization with L-BFGS
    for restart in range(10):
        if time.time() > deadline:
            break
        pts = _init_random()
        res = minimize(_negative_score, pts.flatten(), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < best_score:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_pts = res.x.reshape(-1, 2)

    # Strategy 3: Differential evolution for global exploration (slower but robust)
    if time.time() < deadline - 30:
        bounds = [(-1, 1)] * (2 * N)
        def de_objective(x):
            return _negative_score(x)
        de_res = differential_evolution(de_objective, bounds, maxiter=200,
                                         tol=1e-8, seed=42, polish=False,
                                         popsize=15, init='latinhypercube')
        current_score = -de_res.fun
        if current_score > best_score:
            best_score = current_score
            best_pts = de_res.x.reshape(-1, 2)
            # Polish with L-BFGS
            if time.time() < deadline - 10:
                res = minimize(_negative_score, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_pts = res.x.reshape(-1, 2)

    # Final polish on the best solution
    if best_pts is not None and time.time() < deadline - 5:
        res = minimize(_negative_score, best_pts.flatten(), method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
        current_score = -res.fun
        if current_score > best_score:
            best_score = current_score
            best_pts = res.x.reshape(-1, 2)

    # Ensure validity: clip points strictly inside the unit circle
    if best_pts is not None:
        dists = np.linalg.norm(best_pts, axis=1)
        mask = dists >= 1.0
        if np.any(mask):
            best_pts[mask] *= 0.999 / dists[mask, np.newaxis]

    return best_pts
