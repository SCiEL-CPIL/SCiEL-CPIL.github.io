import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area over all triples of points.
    points: (N, 2) array
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Vectorized calculation of all triangle areas
    # i, j, k distinct indices
    # We can use broadcasting
    # P_i, P_j, P_k
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    
    # To save memory and time, we can loop over i
    min_area = float('inf')
    for i in range(n):
        # Vectors from i to all others
        v1 = points[i+1:] - points[i] # (n-i-1, 2)
        # We need pairs (j, k) from the remaining points
        # This is getting complex to vectorize fully without huge memory.
        # Given N=14, N choose 3 = 364. A simple triple loop is fine.
        pass
        
    # Actually, a triple loop for N=14 is 364 iterations. 
    # 364 * (few flops) is negligible.
    # Let's stick to the simple loop for clarity and low overhead.
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or degenerate, area is 0
        return 0.0

def _score(points):
    """
    Score = min_triangle_area / convex_hull_area
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(params):
    """
    Objective for minimization: -score
    params: flattened array of 2*N coordinates
    """
    points = params.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    timeout = 100.0  # Safety margin for 120s limit
    
    # Initial guess: Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    radius = 1.0
    init_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    init_params = init_points.flatten()
    
    # Bounds: [-2, 2] for each coordinate
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Differential Evolution
    # popsize=10, maxiter=200
    # Should be fast enough.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-4, 
            polish=True
        )
        
        # Check time
        if time.time() - start_time > timeout:
            return init_points
            
        best_points = result.x.reshape(N, 2)
        
    except Exception:
        best_points = init_points
        
    # Optional: Local refinement with Nelder-Mead or L-BFGS-B
    # Since DE finds a good region, a local polish can help.
    # However, the objective is non-smooth. Nelder-Mead is robust.
    from scipy.optimize import minimize
    
    try:
        if time.time() - start_time < timeout - 10:
            res_local = minimize(
                _objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_local.success or _score(res_local.x.reshape(N,2)) > _score(best_points):
                best_points = res_local.x.reshape(N, 2)
    except Exception:
        pass
        
    return best_points
