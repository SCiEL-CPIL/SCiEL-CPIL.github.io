import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _min_area_bary(points_bary):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points_bary: (N, 2) array of (u, v) coordinates. w = 1 - u - v.
    Returns the minimum area.
    """
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Vectorized approach for all triples
    i, j, k = np.indices((N, N, N))
    mask = (i < j) & (j < k)
    i_f = i[mask]
    j_f = j[mask]
    k_f = k[mask]
    
    u_i = u[i_f]; v_i = v[i_f]; w_i = w[i_f]
    u_j = u[j_f]; v_j = v[j_f]; w_j = w[j_f]
    u_k = u[k_f]; v_k = v[k_f]; w_k = w[k_f]
    
    det = (u_i * (v_j * w_k - w_j * v_k) 
         - v_i * (u_j * w_k - w_j * u_k) 
         + w_i * (u_j * v_k - v_j * u_k))
         
    areas = np.abs(det)
    return np.min(areas)

def _project_to_simplex(points_bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial candidates
    inits = []
    
    # 1. Random Dirichlet inits
    for _ in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)[:, :2]
        inits.append(bary)
        
    # 2. Structured Hex-like init (Grid n=3 + Centroid)
    grid_pts = []
    for i in range(4):
        for j in range(4 - i):
            u = (3 - i - j) / 3.0
            v = j / 3.0
            grid_pts.append([u, v])
    grid_pts = np.array(grid_pts)
    centroid = np.array([[1/3, 1/3]])
    hex_init = np.vstack([grid_pts, centroid])
    inits.append(hex_init)
    
    best_area = 0
    best_points = None
    
    # Optimize each init
    for init_bary in inits:
        if time.time() > deadline:
            break
            
        x0 = init_bary.flatten()
        
        def obj(params):
            pb = params.reshape(N, 2)
            pb = _project_to_simplex(pb)
            return -_min_area_bary(pb)
            
        try:
            # Use Nelder-Mead with limited iterations
            res = minimize(
                obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < -best_area:
                pb_opt = res.x.reshape(N, 2)
                pb_opt = _project_to_simplex(pb_opt)
                area = -res.fun
                if area > best_area:
                    best_area = area
                    best_points = pb_opt
                    
        except Exception:
            continue
            
    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Convert to Cartesian
    u = best_points[:, 0]
    v = best_points[:, 1]
    w = 1.0 - u - v
    
    h = np.sqrt(3) / 2
    x = v * 1.0 + w * 0.5
    y = w * h
    
    points_cart = np.column_stack([x, y])
    
    # Final safety clamp
    eps = 1e-9
    for i in range(N):
        x, y = points_cart[i]
        y = max(eps, y)
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x - eps
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) - eps
        x = max(eps, min(1 - eps, x))
        points_cart[i] = [x, y]
        
    return points_cart

if __name__ == "__main__":
    pts = construct()
    x = pts[:, 0]
    y = pts[:, 1]
    h = np.sqrt(3) / 2
    w = y / h
    v = x - w * 0.5
    u = 1 - w - v
    pb = np.column_stack([u, v])
    print("Min Area:", _min_area_bary(pb))
