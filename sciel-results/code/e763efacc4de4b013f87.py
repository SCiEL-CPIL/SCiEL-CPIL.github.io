import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    if points.shape != (N, 2):
        return -np.inf
        
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -np.inf
    except Exception:
        return -np.inf

    # Calculate Minimum Triangle Area
    # Vectorized calculation of all triangle areas
    # Indices for all combinations of 3
    # C(13, 3) = 286
    # We can compute areas using the cross product formula.
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # To be fast, we can loop or vectorize.
    # Vectorizing all 286 at once might be memory heavy if done naively, but 286 is small.
    # Let's just loop for simplicity and robustness, or use a precomputed index list.
    
    min_area = np.inf
    # Precompute indices
    indices = list(itertools.combinations(range(N), 3))
    
    for i, j, k in indices:
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        
        # Vectorized cross product for area
        # 2 * Area = |(p_j - p_i) x (p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale to fit in [0,1] roughly? DE will handle bounds.
    # Let's use bounds [0,1] for x and y.
    # Center the polygon at (0.5, 0.5) and scale to radius 0.5
    init_points = init_points * 0.5 + 0.5
    
    # Objective function for DE (minimization)
    def objective(params):
        # Check time
        if time.time() > deadline:
            return -1e9 # Very bad score
        
        points = params.reshape(N, 2)
        score = _score(points)
        if np.isinf(score):
            return -1e9
        return -score # Maximize score -> Minimize -score

    # Differential Evolution
    # Bounds: each coordinate in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Run DE
    # popsize=20, maxiter=100 should be fast.
    # Let's try popsize=30, maxiter=50 to be safe on time.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=30,
            tol=0.01,
            polish=True # Use Nelder-Mead for final polish
        )
    except Exception:
        # Fallback to initial guess if DE fails
        result = None
    
    if result and result.success:
        best_points = result.x.reshape(N, 2)
    else:
        # If DE didn't succeed or failed, try Nelder-Mead from init
        try:
            res_nm = minimize(objective, init_points.flatten(), method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_nm.success:
                best_points = res_nm.x.reshape(N, 2)
            else:
                best_points = init_points
        except Exception:
            best_points = init_points
            
    # Final check: Ensure the points are valid
    final_score = _score(best_points)
    if np.isinf(final_score):
        best_points = init_points

    return best_points
