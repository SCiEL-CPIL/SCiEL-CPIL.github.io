"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import time


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points using vectorized operations."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            v2 = points[j+1:] - points[i]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull using Andrew's monotone chain."""
    pts = np.array(points)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    hull = np.array(hull)
    
    if len(hull) < 3:
        return 0.0
    
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def _convexity_penalty(points):
    """Return a penalty if points are not strictly convex (in given cyclic order)."""
    n = len(points)
    if n < 3:
        return 1.0
    min_cross = np.inf
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross < min_cross:
            min_cross = cross
    if min_cross <= 0:
        return 1.0 + abs(min_cross)
    return 0.0


def _score_from_params(params):
    """
    Convert perturbation parameters to points and compute score.
    params: array of shape (2*N,)
      - params[0:N] are angular perturbations
      - params[N:2*N] are radial perturbations
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    
    angles = base_angles + params[0:N]
    radii = 1.0 + params[N:2*N]
    
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    penalty = _convexity_penalty(points)
    if penalty > 0:
        return -penalty
    
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area <= 1e-10:
        return -1.0
    
    return min_area / hull_area


def _regular_n_gon(n):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


def construct():
    """
    Construct 14 points in a planar convex region normalized to unit area
    to maximize the minimum triangle area.
    
    Returns:
        np.array of shape (14, 2)
    """
    start_time = time.time()
    time_limit = 110
    
    base_angles = 2 * np.pi * np.arange(N) / N
    
    initial_score = _score_from_params(np.zeros(2*N))
    
    best_params = np.zeros(2*N)
    best_score = initial_score
    
    max_angle_pert = 0.35 * (2 * np.pi / N)
    max_radius_pert = 0.35
    
    bounds = [(-max_angle_pert, max_angle_pert)] * N + [(-max_radius_pert, max_radius_pert)] * N
    
    def objective(params):
        return -_score_from_params(params)
    
    # Stage 1: Multiple DE runs with different seeds
    seeds = [42, 123, 789, 555, 2024]
    for seed in seeds:
        if time.time() - start_time > time_limit * 0.7:
            break
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=40,
                popsize=15,
                tol=1e-6,
                seed=seed,
                polish=False,
                mutation=(0.4, 0.9),
                recombination=0.8
            )
            de_score = -result.fun
            if de_score > best_score:
                best_params = result.x.copy()
                best_score = de_score
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead local refinement
    if time.time() - start_time < time_limit * 0.95:
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_params = nm_result.x.copy()
                best_score = nm_score
        except Exception:
            pass
    
    # Convert best params to points
    angles = base_angles + best_params[0:N]
    radii = 1.0 + best_params[N:2*N]
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]
    
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
    
    return points
# EVOLVE-BLOCK-END
