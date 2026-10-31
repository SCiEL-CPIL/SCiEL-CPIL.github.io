"""Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 13

# EVOLVE-BLOCK-START
import random
import math

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    # Use scipy's ConvexHull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or degenerate
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def construct():
    """
    Represent 13 points as a fixed grid pattern (4x4 minus one corner) and
    optimize the affine transformation to maximize the minimum triangle area ratio.
    
    The base grid is a 4x4 grid of points. We remove one point (e.g., a corner)
    to get 13 points. We then apply an affine transformation (2x2 matrix + translation)
    and optimize the 4 parameters of the 2x2 matrix (translation is irrelevant for
    area ratios since both numerator and denominator scale the same way, and
    translation doesn't affect areas).
    
    Actually, translation doesn't affect triangle areas or hull area, so we only
    need to optimize the 2x2 linear transformation matrix (4 parameters).
    """
    # Base 4x4 grid centered at origin
    base_points = []
    for i in range(4):
        for j in range(4):
            base_points.append([i - 1.5, j - 1.5])
    
    # Remove one point (top-right corner, index 15) to get 13 points
    base_points = [p for idx, p in enumerate(base_points) if idx != 15]
    base_points = np.array(base_points)
    
    # Optimize the 2x2 affine matrix [[a, b], [c, d]]
    # The transformed points are: base_points @ M.T
    # where M = [[a, c], [b, d]] (we'll define it carefully)
    
    def objective(params):
        a, b, c, d = params
        M = np.array([[a, b], [c, d]])
        transformed = base_points @ M.T
        return -_score(transformed)
    
    # Bounds for the affine parameters
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=500,
        tol=1e-6,
        popsize=15,
        init='sobol',
        polish=True,
        update_interval=10
    )
    
    a, b, c, d = result.x
    M = np.array([[a, b], [c, d]])
    points = base_points @ M.T
    
    # Refine with local optimization if needed
    # Use Nelder-Mead for local refinement
    try:
        local_result = minimize(
            objective,
            x0=result.x,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if local_result.fun < result.fun:
            a, b, c, d = local_result.x
            M = np.array([[a, b], [c, d]])
            points = base_points @ M.T
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
