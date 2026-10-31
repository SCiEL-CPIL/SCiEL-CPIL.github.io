import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = 0.5 * 1.0 * (SQRT3 / 2.0) # sqrt(3)/4

def min_triangle_area(points):
    """
    Calculate the minimum area of all triangles formed by the points using vectorization.
    """
    n = points.shape[0]
    # Generate indices for all unique triplets (i, j, k) with i < j < k
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Vectorized cross product calculation for 2D
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    # Distance to sides:
    # Side 1: y = 0 -> dist = y
    # Side 2: y = sqrt(3)*x -> dist = (sqrt(3)*x - y) / sqrt(1+3) = (sqrt(3)*x - y)/2 ? 
    # Actually, line is sqrt(3)x - y = 0. Distance is |sqrt(3)x - y| / 2.
    # Side 3: y = -sqrt(3)(x - 1) = -sqrt(3)x + sqrt(3) -> sqrt(3)x + y - sqrt(3) = 0.
    # Distance is |sqrt(3)x + y - sqrt(3)| / 2.
    
    d1 = y
    d2 = (SQRT3 * x - y) / 2.0
    d3 = (SQRT3 * x + y - SQRT3) / 2.0
    
    # We want points inside, so d1, d2, d3 should be >= 0.
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = min_triangle_area(points)
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Pre-compute indices for speed? No, min_triangle_area does it every time.
    # We can cache indices.
    n = N
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    def fast_min_area(points):
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        dx_ji = p_j[:, 0] - p_i[:, 0]
        dy_ji = p_j[:, 1] - p_i[:, 1]
        dx_ki = p_k[:, 0] - p_i[:, 0]
        dy_ki = p_k[:, 1] - p_i[:, 1]
        cross = dx_ji * dy_ki - dy_ji * dx_ki
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def fast_objective(flat_params):
        points = flat_params.reshape(N, 2)
        min_area = fast_min_area(points)
        x = points[:, 0]
        y = points[:, 1]
        d1 = y
        d2 = (SQRT3 * x - y) / 2.0
        d3 = (SQRT3 * x + y - SQRT3) / 2.0
        pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
        return -min_area + 1000.0 * pen

    # Initializations
    inits = []
    
    # 1. Random uniform
    for seed in range(20):
        rng = np.random.RandomState(seed)
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
        inits.append(init_points.flatten())
        
    # 2. Structured: Vertices, Midpoints, Centroid, + 4 random
    vertices = np.array([[0, 0], [1, 0], [0.5, SQRT3/2]])
    mids = np.array([[0.5, 0], [0.75, SQRT3/4], [0.25, SQRT3/4]])
    centroid = np.array([[0.5, SQRT3/6]])
    base_struct = np.vstack([vertices, mids, centroid]) # 7 points
    
    for seed in range(10):
        rng = np.random.RandomState(seed + 100)
        extra = np.zeros((4, 2))
        for i in range(4):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            extra[i, 0] = v * 1.0 + w * 0.5
            extra[i, 1] = w * SQRT3 / 2.0
        init_points = np.vstack([base_struct, extra])
        # Shuffle to avoid symmetry bias?
        idx = rng.permutation(N)
        init_points = init_points[idx]
        inits.append(init_points.flatten())

    # Run optimization on each init
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = minimize(
                fast_objective, 
                init, 
                method='Powell', 
                options={'maxiter': 200, 'maxfev': 5000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = fast_min_area(opt_points)
            x = opt_points[:, 0]
            y = opt_points[:, 1]
            d1 = y
            d2 = (SQRT3 * x - y) / 2.0
            d3 = (SQRT3 * x + y - SQRT3) / 2.0
            pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    # Final polish with Basin Hopping if time remains
    if time.time() - start_time < time_limit - 10:
        try:
            bh_result = basinhopping(
                fast_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 100, 'maxfev': 2000},
                niter=20,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = fast_min_area(bh_points)
            x = bh_points[:, 0]
            y = bh_points[:, 1]
            d1 = y
            d2 = (SQRT3 * x - y) / 2.0
            d3 = (SQRT3 * x + y - SQRT3) / 2.0
            bh_pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    return best_points
