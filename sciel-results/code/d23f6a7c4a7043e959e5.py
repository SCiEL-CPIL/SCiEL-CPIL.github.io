import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11

# Pre-compute indices for all triangles
indices = np.array(list(itertools.combinations(range(N), 3)))

def get_min_area_vectorized(points):
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    det = (p1[:, 0] - p3[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p1[:, 1] - p3[:, 1]) * (p2[:, 0] - p3[:, 0])
    areas = 0.5 * np.abs(det)
    return np.min(areas)

def is_inside_triangle_vectorized(points):
    x = points[:, 0]
    y = points[:, 1]
    h = np.sqrt(3)
    # y >= 0
    # y <= sqrt(3) x
    # y <= sqrt(3) (1-x)
    d1 = -y
    d2 = y - h * x
    d3 = y - h * (1 - x)
    
    v1 = np.maximum(0, d1)
    v2 = np.maximum(0, d2)
    v3 = np.maximum(0, d3)
    
    return np.sum(v1**2 + v2**2 + v3**2)

def cost_function(params):
    points = params.reshape(-1, 2)
    penalty = is_inside_triangle_vectorized(points)
    if penalty > 1e-6:
        return 100 + penalty
    a = get_min_area_vectorized(points)
    return -a

def construct():
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Differential Evolution
    try:
        res = differential_evolution(cost_function, bounds, maxiter=50, popsize=20, tol=1e-4, seed=42, polish=False)
        best_params = res.x
    except Exception:
        # Fallback to a simple grid if DE fails
        # Grid of 11 points?
        # Just return random valid points
        best_params = np.random.uniform(0.1, 0.9, (N, 2))
        # Ensure they are inside
        for i in range(N):
            x, y = best_params[i]
            if y > np.sqrt(3)*x or y > np.sqrt(3)*(1-x) or y < 0:
                best_params[i] = [0.5, 0.1]
                
    # Local Refinement
    def local_cost(params):
        points = params.reshape(-1, 2)
        penalty = is_inside_triangle_vectorized(points)
        if penalty > 1e-6:
            return 100 + penalty
        a = get_min_area_vectorized(points)
        return -a

    try:
        res_local = minimize(local_cost, best_params, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        best_params = res_local.x
    except Exception:
        pass
        
    points = best_params.reshape(-1, 2)
    
    # Final validation
    # If points are outside, project them or replace
    # For robustness, check each point
    final_points = []
    for p in points:
        x, y = p
        if y < 0: y = 0
        if x < 0: x = 0
        if x > 1: x = 1
        if y > np.sqrt(3)*x:
            # Project onto line y = sqrt(3)x
            # This is complex, just clamp
            # If outside, move to center
            if x < 0.5:
                 y = np.sqrt(3)*x * 0.9
            else:
                 y = np.sqrt(3)*(1-x) * 0.9
        if y > np.sqrt(3)*(1-x):
             if x >= 0.5:
                 y = np.sqrt(3)*(1-x) * 0.9
             else:
                 y = np.sqrt(3)*x * 0.9
        final_points.append([x, y])
        
    return np.array(final_points)
