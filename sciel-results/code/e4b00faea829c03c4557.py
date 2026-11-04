"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns the score (higher is better).
    """
    points = np.asarray(points)
    n = len(points)
    if n < 3:
        return 0.0
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # 2. Calculate Minimum Triangle Area
    # There are C(13, 3) = 286 triangles
    min_area = float('inf')
    
    # Vectorized calculation for all triples
    # We can compute all triangle areas at once using cross product
    # Triangle area = 0.5 * |cross(p2-p1, p3-p1)|
    
    # To avoid O(N^3) memory explosion for large N, but N=13 is small.
    # C(13,3) = 286.
    # We can just loop or use a small vectorized approach.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Cross product of (p2-p1) and (p3-p1)
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                if min_area < 1e-12:
                    return 0.0
                    
    if min_area == float('inf') or min_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area


def _negative_score(points):
    """
    Objective function for minimization.
    """
    return -_score(points)


def _regular_polygon(n, radius=1.0, rotation=0.0):
    """
    Generate vertices of a regular n-gon.
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    """
    Construct 13 points to maximize the minimum triangle area relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular polygon
    initial_points = _regular_polygon(N)
    current_score = _score(initial_points)
    best_score = current_score
    best_points = initial_points.copy()
    
    # Multi-start Nelder-Mead optimization
    # We try several random perturbations of the regular polygon
    # and optimize each one.
    
    rng = np.random.RandomState(42)
    
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x0 = initial_points
        else:
            # Random perturbation
            # Scale is important. Regular polygon has radius 1.
            # Perturbations should be small enough to stay near the shape but large enough to escape local minima.
            x0 = initial_points + rng.normal(0, 0.1, size=(N, 2))
        
        # Optimize
        try:
            result = minimize(_negative_score, x0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.success or result.fun < -best_score:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x
        except Exception:
            continue
            
    # One more pass with the best point found so far, maybe with tighter tolerance or different start
    # If we have time, do a final polish
    if time.time() < deadline - 5.0 and best_points is not None:
        try:
            result = minimize(_negative_score, best_points, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
