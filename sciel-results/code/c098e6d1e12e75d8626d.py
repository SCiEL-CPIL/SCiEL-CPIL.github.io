import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    """
    n = points.shape[0]
    # Create indices for all unique triplets (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Vectorized cross product magnitude
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = np.min(triangle_areas_vectorized(points))
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6
    return objective(flat_params)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Define bounds for L-BFGS-B
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, SQRT3/2)] * N
    
    def get_symmetric_init(seed):
        """
        Generate a symmetric initial guess to help convergence.
        """
        rng = np.random.RandomState(seed)
        # 11 points:
        # 1 at centroid
        # 3 near vertices (slightly inside)
        # 3 near midpoints of edges
        # 3 in a small inner triangle
        # 1 extra? 
        # Let's just use random points but ensure some symmetry by mirroring?
        # Simpler: Random points, but we'll run many restarts.
        
        # For better results, we can try a grid-based init
        pts = []
        # Centroid
        pts.append([0.5, SQRT3/6])
        
        # 3 points at 1/3 height, 1/3 width from left/right?
        # Let's use a known good configuration structure:
        # Rows of points.
        # Row 0 (base): 4 points
        # Row 1: 3 points
        # Row 2: 2 points
        # Row 3: 1 point
        # Total 10? Need 11.
        # Maybe 4, 3, 2, 2?
        
        # Let's stick to random restarts with L-BFGS-B which is robust.
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        return init_points

    # Run multiple restarts with L-BFGS-B
    num_restarts = 500 # Increase restarts as L-BFGS-B is faster
    
    for seed in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        init_points = get_symmetric_init(seed)
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                safe_objective, 
                flat_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfev': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas_vectorized(opt_points))
            pen = penalty(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    # Final polish with basinhopping on the best point found
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            bh_result = basinhopping(
                safe_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'maxiter': 500, 'maxfev': 5000},
                niter=50, # Increase iterations for polish
                seed=123,
                stepsize=0.5
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(triangle_areas_vectorized(bh_points))
            bh_pen = penalty(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
