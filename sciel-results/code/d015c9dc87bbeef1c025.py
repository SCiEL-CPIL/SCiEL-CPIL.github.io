"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            # Vectorized over k > j
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                # Cross product in 2D
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                min_area = min(min_area, np.min(areas))
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective_full(x):
    """
    Objective function for full coordinate optimization.
    x: (2*N,) array.
    """
    points = x.reshape(N, 2)
    return -_score(points)

def _objective_radial(r):
    """
    Objective function for radial optimization (fixed regular angles).
    r: (N,) array of radii.
    """
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Regular polygon initialization
    angles = 2 * np.pi * np.arange(N) / N
    init_r = np.ones(N)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 1. Radial Optimization (14D)
    # This exploits the likely symmetry of the optimal solution
    if time.time() < deadline:
        try:
            res_r = minimize(_objective_radial, init_r, method='Nelder-Mead',
                             options={'maxiter': 5000, 'maxfev': 5000})
            score_r = -res_r.fun
            points_r = np.column_stack([res_r.x * np.cos(angles), res_r.x * np.sin(angles)])
            if score_r > best_score:
                best_score = score_r
                best_points = points_r
        except Exception:
            pass

    # 2. Full Coordinate Optimization (28D)
    # Refine the best solution found so far
    if best_points is None:
        start_pts = init_points
    else:
        start_pts = best_points
        
    if time.time() < deadline:
        try:
            res_f = minimize(_objective_full, start_pts.flatten(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'maxfev': 5000})
            score_f = -res_f.fun
            points_f = res_f.x.reshape(N, 2)
            if score_f > best_score:
                best_score = score_f
                best_points = points_f
        except Exception:
            pass
            
    # 3. Additional Restart with Perturbation
    # Try to escape local optima with a different starting point
    if time.time() < deadline - 5:
        try:
            rng = np.random.RandomState(42)
            perturbed = init_points + rng.normal(0, 0.2, init_points.shape)
            res_p = minimize(_objective_full, perturbed.flatten(), method='Nelder-Mead',
                             options={'maxiter': 3000, 'maxfev': 3000})
            score_p = -res_p.fun
            points_p = res_p.x.reshape(N, 2)
            if score_p > best_score:
                best_score = score_p
                best_points = points_p
        except Exception:
            pass

    if best_points is None:
        return init_points
        
    return best_points
# EVOLVE-BLOCK-END
