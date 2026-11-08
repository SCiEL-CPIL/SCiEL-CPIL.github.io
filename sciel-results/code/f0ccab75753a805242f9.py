"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import itertools

N = 14
DEADLINE_MARGIN = 5.0  # seconds before hard kill to return


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are degenerate, use shoelace on sorted points
        # This shouldn't happen with our initializations
        return 1e-10


def _score(pts):
    """Score = min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _negative_score(pts_flat):
    """Objective for minimization: negative of the score."""
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _jittered_polygon(n, radius=1.0, jitter=0.1, seed=42):
    """Generate a regular n-gon with radial jitter."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = radius + rng.uniform(-jitter, jitter, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_points(n, seed=42):
    """Generate random points in a unit square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n, 2))


def _optimize_from_start(pts0, time_budget):
    """Run Nelder-Mead optimization from a starting point with time budget."""
    start_time = time.time()
    pts_flat = pts0.flatten()
    
    # First try Powell for a rough convergence
    try:
        res_powell = minimize(
            _negative_score,
            pts_flat,
            method='Powell',
            options={'maxiter': 50, 'ftol': 1e-8, 'xtol': 1e-8}
        )
        elapsed = time.time() - start_time
        if elapsed < time_budget * 0.5:
            pts_flat = res_powell.x
    except Exception:
        pass
    
    # Then Nelder-Mead polish with tight tolerances
    try:
        res_nm = minimize(
            _negative_score,
            pts_flat,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
        )
        pts_flat = res_nm.x
    except Exception:
        pass
    
    return pts_flat.reshape(N, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_pts = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon (best theoretical starting point)
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # Jittered polygons with different seeds
    for seed in range(10):
        initial_configs.append(_jittered_polygon(N, radius=1.0, jitter=0.05, seed=seed))
    
    # Random points
    for seed in range(5):
        initial_configs.append(_random_points(N, seed=seed))
    
    # Optimize from each starting point with time budget
    for i, pts0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Allocate remaining time budget
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Give more time to earlier (likely better) starts
        if i < 3:
            time_budget = min(remaining, 10.0)
        elif i < 8:
            time_budget = min(remaining, 5.0)
        else:
            time_budget = min(remaining, 3.0)
        
        try:
            pts_opt = _optimize_from_start(pts0, time_budget)
        except Exception:
            continue
        
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_pts = pts_opt
        
        # If we've found a very good score, we can stop early
        if best_score > 0.05:
            # Check if we have time for a couple more quick refinements
            if time.time() > deadline - 10:
                break
    
    # If no optimization succeeded, fall back to regular polygon
    if best_pts is None:
        best_pts = _regular_polygon(N, radius=1.0)
    
    # Final quick polish if we have time
    if time.time() < deadline - 2.0:
        try:
            pts_flat = best_pts.flatten()
            res = minimize(
                _negative_score,
                pts_flat,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
            )
            final_pts = res.x.reshape(N, 2)
            final_score = _score(final_pts)
            if final_score > _score(best_pts):
                best_pts = final_pts
        except Exception:
            pass
    
    return best_pts
