import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score_from_points(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    min_area = float('inf')
    n = len(points)
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        if area < min_area:
            min_area = area
            
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _angular_objective(angles):
    """
    Objective function for differential evolution.
    angles: array of 13 angles.
    Returns negative score for minimization.
    """
    angles = np.asarray(angles)
    # Points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    score = _score_from_points(points)
    return -score

def construct():
    # Bounds for angles: [0, 2*pi)
    # We can sort angles to avoid permutations, but DE doesn't know about sorting.
    # However, the objective function is permutation invariant.
    # To help convergence, we can sort the angles inside the objective?
    # No, sorting makes the function non-smooth.
    # Instead, we rely on the fact that the global optimum is a set of angles.
    # DE will find some permutation of them.
    
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    # Differential Evolution settings
    # maxiter and popsize need to be balanced with the 120s limit.
    # N=13 is small.
    res = differential_evolution(
        _angular_objective, 
        bounds=bounds, 
        seed=42, 
        maxiter=200, 
        popsize=15,
        tol=1e-5,
        polish=True  # Use Nelder-Mead to polish the result
    )
    
    best_angles = res.x
    # Sort angles to get a consistent ordering (optional, but good for determinism)
    best_angles = np.sort(best_angles)
    
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.column_stack((x, y))
    
    return points
# EVOLVE-BLOCK-END
