"""Heilbronn convex regions: place 13 points to maximize minimum triangle area / convex hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized computation for speed
    # For each pair (i,j), compute cross products with all k
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            # Cross product of v1 with (points[k] - points[i]) for all k != i,j
            diffs = points - points[i]
            crosses = v1[0] * diffs[:, 1] - v1[1] * diffs[:, 0]
            # Exclude i and j (they give zero)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(mask):
                areas = 0.5 * np.abs(crosses[mask])
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on convex hull
        # This shouldn't happen with valid points
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(x):
    """Objective function for optimization (minimize negative score)."""
    points = x.reshape(N, 2)
    score = _score(points)
    return -score

def _initial_points():
    """Generate initial points using a structured approach."""
    # Start with points on a circle and some interior points
    # Use a regular polygon-like structure
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with a good initial configuration
    best_points = _initial_points()
    best_score = _score(best_points)
    
    # Use differential evolution for global optimization
    # Flatten the points into a 1D array
    x0 = best_points.flatten()
    
    # Bounds: points should be in [0, 1] x [0, 1] roughly
    bounds = [(0, 1)] * (2 * N)
    
    try:
        # Run differential evolution
        result = differential_evolution(
            _objective,
            bounds,
            x0=x0,
            maxiter=200,
            tol=1e-8,
            seed=42,
            polish=True,
            popsize=15,
            init='sobol'
        )
        
        if result.success or result.fun < 0:
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        
    except Exception:
        pass
    
    # Check if we still have time for local refinement
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            # Local refinement using L-BFGS-B
            x_start = best_points.flatten()
            result_local = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result_local.fun < 0:
                candidate_points = result_local.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
                    
        except Exception:
            pass
    
    # Normalize: ensure points are in a reasonable range
    # The score is invariant to affine transformations that preserve area ratios,
    # but let's make sure the points are well-scaled
    
    return best_points

# EVOLVE-BLOCK-END
