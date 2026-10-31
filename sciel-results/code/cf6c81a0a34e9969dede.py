"""Heilbronn convex regions – 13 points, maximize min triangle area / convex hull area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds to leave before the kill

# Precompute all C(13,3)=286 triples
TRIPLES = list(combinations(range(N), 3))
TRIPLES_ARR = np.array(TRIPLES, dtype=np.int32)  # shape (286, 3)


def _min_triangle_area(points):
    """Minimum area of any triangle formed by triples of the given points."""
    # points: (13, 2)
    # For each triple (i, j, k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    pi = points[TRIPLES_ARR[:, 0]]   # (286, 2)
    pj = points[TRIPLES_ARR[:, 1]]
    pk = points[TRIPLES_ARR[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _convex_hull_area(points):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # in 2D, .volume is the area
    except Exception:
        # Fallback: shoelace on sorted points
        idx = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[idx]
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _objective(x):
    """Objective to minimize (negative score). x is flat (13*2,) array."""
    points = x.reshape(N, 2)
    # Penalize if hull area is too small
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-6:
        return 1.0
    min_area = _min_triangle_area(points)
    return -min_area / hull_area


def _refine(points, time_budget):
    """Local refinement using Nelder-Mead / BFGS on the flat parameter vector."""
    x0 = points.flatten()
    best_x = x0.copy()
    best_val = _objective(x0)
    
    start = time.time()
    while time.time() - start < time_budget:
        # Try Nelder-Mead
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # Try BFGS
        try:
            res = minimize(_objective, best_x, method='BFGS',
                           options={'maxiter': 500, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # Check if we've converged (no improvement)
        current_val = _objective(best_x)
        if abs(current_val - best_val) < 1e-14:
            break
    
    return best_x.reshape(N, 2)


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    
    # Initial guess: regular 13-gon on unit circle
    angles = 2 * np.pi * np.arange(N) / N
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = initial_points.copy()
    best_score = _score(initial_points)
    
    # Also try a few other initial configurations
    # Configuration 2: two concentric rings
    n_outer = 7
    n_inner = 6
    angles_outer = 2 * np.pi * np.arange(n_outer) / n_outer
    angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + np.pi / n_inner
    r_outer = 1.0
    r_inner = 0.4
    pts2 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    s2 = _score(pts2)
    if s2 > best_score:
        best_score = s2
        best_points = pts2.copy()
    
    # Configuration 3: slightly perturbed regular polygon
    rng = np.random.RandomState(42)
    for trial in range(3):
        pert = rng.randn(N, 2) * 0.1
        pts3 = initial_points + pert
        s3 = _score(pts3)
        if s3 > best_score:
            best_score = s3
            best_points = pts3.copy()
    
    # Differential Evolution
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 30:  # Need at least 30s for DE + refinement
        # Bounds: points in [-1.5, 1.5] x [-1.5, 1.5]
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        de_start = time.time()
        de_budget = min(remaining * 0.6, 60.0)  # Use up to 60% of remaining, max 60s
        
        try:
            result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='sobol',
                updating='immediate',
                callback=lambda xk, convergence: (
                    time.time() - de_start > de_budget
                )
            )
            if result.fun < _objective(best_points.flatten()):
                best_points = result.x.reshape(N, 2)
                best_score = -result.fun
        except Exception:
            pass
    
    # Local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining > 5:
        refined = _refine(best_points, remaining - 1.0)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    # Final check: ensure we return within deadline
    return best_points
