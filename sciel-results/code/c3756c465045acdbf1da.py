import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.array(points)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for degeneracy (duplicate points)
    # If any two points are identical, min area is 0
    # We can check min distance or just rely on hull/triangle calc returning 0
    
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Compute Min Triangle Area
    # Vectorized calculation of all triangle areas
    # Number of triangles: C(13, 3) = 286
    # We can loop or vectorize. Looping 286 times is fast.
    min_area = float('inf')
    
    # Indices for all combinations of 3
    # Using itertools.combinations is cleaner but slightly slower than numpy indexing if not careful.
    # For N=13, simple loops are fine.
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, N):
                pk = points[k]
                v2 = pk - pi
                # Area = 0.5 * |cross(v1, v2)|
                # cross(v1, v2) = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0

    if min_area == float('inf'):
        return 0.0

    return min_area / hull_area

def _neg_score(params):
    """
    Negative score for minimization.
    """
    points = params.reshape(N, 2)
    return -_calculate_score(points)

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin before 120s/150s kill
    
    # Initial bounds: [-1, 1] for x and y
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    # Differential Evolution
    # popsize=15, maxiter=200 is a good balance for speed/quality in 26D
    # tol=0.01, mutation=(0.5, 1.5)
    try:
        res = differential_evolution(
            _neg_score, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=0.01, 
            popsize=15,
            polish=False, # We will do our own polish
            init='sobol' # Better coverage than 'latinhypercube' or 'random'
        )
        
        best_params = res.x
        best_score = -res.fun
        
    except Exception as e:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_params = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
        best_score = _calculate_score(best_params.reshape(N,2))

    # Local Refinement
    if time.time() - start_time < deadline - 20:
        try:
            # Use BFGS or L-BFGS-B
            # L-BFGS-B is robust for box constraints, but we don't strictly need them if we started inside.
            # However, points can drift. Let's use BFGS without bounds first, 
            # or L-BFGS-B with loose bounds.
            
            res_local = minimize(
                _neg_score, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < -best_score:
                best_params = res_local.x
                best_score = -res_local.fun
                
        except Exception:
            pass
            
    # One more polish if time permits
    if time.time() - start_time < deadline - 5:
        try:
            res_local2 = minimize(
                _neg_score, 
                best_params, 
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            if res_local2.fun < -best_score:
                best_params = res_local2.x
                best_score = -res_local2.fun
        except Exception:
            pass

    return best_params.reshape(N, 2)
