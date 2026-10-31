import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    """
    n = points.shape[0]
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    # Inequalities for being inside the triangle:
    # 1. y >= 0
    # 2. sqrt(3)*x - y >= 0  => y <= sqrt(3)*x
    # 3. sqrt(3)*(1-x) - y >= 0 => y <= sqrt(3)*(1-x)
    
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 * (1.0 - x) - y
    
    # Quadratic penalty for violations
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def smooth_min_cost(points, k):
    """
    Compute the negative smooth minimum of the triangle areas.
    We want to minimize this, which maximizes the smooth minimum area.
    soft_min(a) = -1/k * log(sum(exp(-k*a)))
    cost = -soft_min(a) = 1/k * log(sum(exp(-k*a)))
    """
    areas = triangle_areas_vectorized(points)
    # Scale areas to avoid overflow in exp
    # Max area is roughly 0.433. k can be large.
    # To be safe, we can shift areas or just use moderate k.
    # Let's use a stable log-sum-exp implementation.
    max_area = np.max(areas)
    # If all areas are 0, return 0
    if max_area == 0:
        return 0.0
    
    # We want min(areas).
    # Let's compute -1/k * log(sum(exp(-k * areas)))
    # This is equivalent to log(sum(exp(-k*(areas - min_area)))) / (-k) + min_area
    # Actually, simpler:
    # LSE(z) = log(sum(exp(z)))
    # We want LSE(-k * areas)
    # To be stable: max_z = max(-k * areas) = -k * min(areas)
    # LSE(-k * areas) = max_z + log(sum(exp(-k * areas - max_z)))
    
    z = -k * areas
    max_z = np.max(z)
    if not np.isfinite(max_z):
        return 1e6
    lse = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -lse / k
    return -soft_min_val # We want to minimize -soft_min, so cost = -soft_min

def objective(flat_params, k):
    points = flat_params.reshape(N, 2)
    cost = smooth_min_cost(points, k)
    pen = penalty(points)
    return cost + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Helper to generate random points in the triangle
    def random_points(rng):
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            pts[i, 0] = v * 1.0 + w * 0.5
            pts[i, 1] = w * SQRT3 / 2.0
        return pts

    # We will use L-BFGS-B with a smooth approximation.
    # We'll increase 'k' (sharpness) gradually or use a fixed high 'k'.
    # A fixed high 'k' (e.g., 50-100) works well with L-BFGS-B if initialized well.
    
    num_restarts = 30 # Can be higher if fast
    k_value = 50.0 # Sharpness of the soft-min
    
    for seed in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        rng = np.random.RandomState(seed)
        init_points = random_points(rng)
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                objective, 
                flat_init, 
                args=(k_value,),
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            # Clip points to be strictly inside or on boundary to avoid numerical issues in final eval
            # The penalty should handle this, but let's ensure no NaNs
            if not np.all(np.isfinite(opt_points)):
                continue
                
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

    # Refine the best point found with a higher k or exact min using Powell?
    # Powell is good for polishing non-smooth functions.
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            # Use Powell to polish the result, using the exact (non-smooth) objective
            def exact_objective(flat_params):
                points = flat_params.reshape(N, 2)
                min_area = np.min(triangle_areas_vectorized(points))
                pen = penalty(points)
                return -min_area + 1000.0 * pen

            bh_result = minimize(
                exact_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 50000, 'disp': False}
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
