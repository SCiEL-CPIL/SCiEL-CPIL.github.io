"""Improved program using Differential Evolution for global search."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _compute_score_flat(flat_points):
    """
    Compute the negative score for a flat array of 2N points.
    Minimizes negative score (maximizes score).
    """
    points = flat_points.reshape(N, 2)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # If points are collinear or fewer than 3, return a bad score (high positive for minimization)
        return 1000.0 
        
    if hull_area < 1e-10:
        return 1000.0

    # Compute min triangle area
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < min_area:
                    min_area = area
                    
    score = min_area / hull_area
    return -score

def construct():
    """
    Place 14 points to maximize the minimum triangle area divided by the convex hull area.
    Uses Differential Evolution for global optimization, followed by Nelder-Mead for local refinement.
    """
    # Bounds for each coordinate
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # popsize=20, maxiter=100 is a reasonable balance for 28 dimensions within 120s
    result = differential_evolution(
        _compute_score_flat, 
        bounds, 
        popsize=20, 
        maxiter=100, 
        tol=1e-5,
        seed=42,
        polish=False 
    )
    
    best_points = result.x.reshape(N, 2)
    
    # Local refinement with Nelder-Mead
    res = minimize(
        _compute_score_flat, 
        best_points.ravel(), 
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    if res.fun < result.fun:
        best_points = res.x.reshape(N, 2)
        
    return best_points
# EVOLVE-BLOCK-END
