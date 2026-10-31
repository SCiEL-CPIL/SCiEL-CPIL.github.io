"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from scipy.optimize import Bounds

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: minimum triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return -1.0
    return min_area / hull_area

def _negative_score(points_flat):
    """Objective function for optimization (negative score to minimize)."""
    points = points_flat.reshape((N, 2))
    score = _score(points)
    if score < 0:
        return 1e6  # Large penalty for invalid configurations
    return -score

def construct():
    """Construct 13 points to maximize the minimum triangle area relative to convex hull area."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds margin
    
    best_score = 0.0
    best_points = None
    
    # Initialize with a regular polygon-like configuration
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = initial_points.copy()
    best_score = _score(initial_points)
    
    # Define bounds for each coordinate (normalized to [-1, 1] for optimization)
    bounds = [(-1, 1)] * (2 * N)
    
    # Run differential evolution for global optimization
    try:
        result = differential_evolution(
            _negative_score,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False,
            update_interval=10,
            callback=lambda x, convergence: (
                time.time() - start_time > deadline - 20
            )
        )
        
        if result.success or result.fun < 0:
            candidate_points = result.x.reshape((N, 2))
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
    except Exception:
        pass
    
    # Local refinement using Nelder-Mead or BFGS
    remaining_time = deadline - 20 - (time.time() - start_time)
    if remaining_time > 5 and best_points is not None:
        try:
            result_local = minimize(
                _negative_score,
                x0=best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result_local.fun < 0:
                candidate_points = result_local.x.reshape((N, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    remaining_time = deadline - 5 - (time.time() - start_time)
    if remaining_time > 2 and best_points is not None:
        try:
            result_polish = minimize(
                _negative_score,
                x0=best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            
            if result_polish.fun < 0:
                candidate_points = result_polish.x.reshape((N, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Ensure we return a valid point set
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
