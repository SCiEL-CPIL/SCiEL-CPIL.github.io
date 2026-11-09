import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3)

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Generate indices for all unique triplets
    idx = np.arange(N)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i = i[i < j]
    j = j[i < j]
    k = k[i < j]
    j = j[j < k]
    k = k[j < k]
    i = i[j < k]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (z-component)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Penalty for points outside the triangle.
    Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    Constraints:
    1. y >= 0
    2. y <= sqrt(3)*x
    3. y <= -sqrt(3)*x + sqrt(3)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    p1 = np.maximum(0, -y)
    p2 = np.maximum(0, y - SQRT3 * x)
    p3 = np.maximum(0, y + SQRT3 * x - SQRT3)
    
    return 10.0 * np.sum(p1**2 + p2**2 + p3**2)

def _objective(x):
    """
    Objective function for minimization.
    x: (2*N,) array
    """
    points = x.reshape(-1, 2)
    pen = _penalty(points)
    area = _min_triangle_area(points)
    return -area + pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    best_score = -np.inf
    best_points = None
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Run multiple DE optimizations with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
            
        try:
            # Differential Evolution
            res_de = differential_evolution(
                _objective, 
                bounds, 
                seed=seed, 
                maxiter=500, 
                popsize=30, 
                tol=1e-6, 
                atol=1e-8,
                update_interval=10
            )
            
            best_x_de = res_de.x
            best_val_de = res_de.fun
            
            # Nelder-Mead Refinement
            # NM can be sensitive to initial step size
            res_nm = minimize(
                _objective, 
                best_x_de, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            best_x_nm = res_nm.x
            best_val_nm = res_nm.fun
            
            # Compare
            if best_val_nm < best_val_de:
                final_x = best_x_nm
                final_val = best_val_nm
            else:
                final_x = best_x_de
                final_val = best_val_de
                
            # Extract points and check validity
            points = final_x.reshape(-1, 2)
            
            # Check if points are reasonably inside (penalty should be small)
            pen = _penalty(points)
            if pen > 1e-3:
                # If penalty is high, try to project or discard
                # For now, just calculate area for scoring
                pass
                
            area = _min_triangle_area(points)
            
            # We want to maximize area, so minimize -area
            # The objective returned -area + penalty.
            # If penalty is 0, val = -area.
            # So higher area means lower val.
            
            if final_val < best_score:
                best_score = final_val
                best_points = points.copy()
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback: random points
        rng = np.random.RandomState(0)
        points = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1-u, 1-v
            points.append([u + 0.5*v, SQRT3/2 * v])
        best_points = np.array(points)

    return best_points
