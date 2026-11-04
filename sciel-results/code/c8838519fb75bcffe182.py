import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle indices
tri_indices = []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def min_area_vectorized(pts):
    """
    Compute minimum triangle area for N points, vectorized.
    pts: (N, 2) array
    """
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def bary_to_cart(bary_pts):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    P = u*A + v*B + w*C
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    w = 1.0 - u - v
    
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2) * w
    
    return np.column_stack((x, y))

def project_bary(bary_pts):
    """
    Project barycentric coordinates (u, v) to the valid triangle region.
    """
    u = bary_pts[:, 0].copy()
    v = bary_pts[:, 1].copy()
    
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack((u, v))

def objective_flat(x):
    """
    Objective function for Nelder-Mead: minimize negative min area.
    x: flat array of length 2*N (Cartesian)
    """
    pts = x.reshape(N, 2)
    area = min_area_vectorized(pts)
    return -area

def sa_optimize(rng, n_iterations=20000, t_start=0.005, t_end=1e-6):
    """
    Simulated Annealing to find 11 points in barycentric coords.
    """
    u = np.sqrt(rng.random(N))
    v = (1 - u) * rng.random(N)
    
    bary_pts = np.column_stack((u, v))
    cart_pts = bary_to_cart(bary_pts)
    current_area = min_area_vectorized(cart_pts)
    
    best_bary = bary_pts.copy()
    best_area = current_area
    
    alpha = (t_end / t_start) ** (1.0 / n_iterations)
    temp = t_start
    step_size = 0.05
    
    for i in range(n_iterations):
        idx = rng.integers(0, N)
        
        du = rng.normal(0, step_size)
        dv = rng.normal(0, step_size)
        
        new_u = bary_pts[idx, 0] + du
        new_v = bary_pts[idx, 1] + dv
        
        new_u = max(0.0, new_u)
        new_v = max(0.0, new_v)
        if new_u + new_v > 1.0:
            s = 1.0 / (new_u + new_v)
            new_u *= s
            new_v *= s
            
        old_u = bary_pts[idx, 0]
        old_v = bary_pts[idx, 1]
        
        bary_pts[idx, 0] = new_u
        bary_pts[idx, 1] = new_v
        
        cart_pts = bary_to_cart(bary_pts)
        new_area = min_area_vectorized(cart_pts)
        
        delta = new_area - current_area
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_bary = bary_pts.copy()
        else:
            bary_pts[idx, 0] = old_u
            bary_pts[idx, 1] = old_v
            
        temp *= alpha
        step_size *= 0.9999
        
    return best_bary, best_area

def construct():
    """
    Main function: find 11 points in the unit triangle maximizing minimum triangle area.
    """
    rng = np.random.default_rng(42)
    
    best_points_cart = None
    best_score = -np.inf
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        best_bary, sa_score = sa_optimize(rng, n_iterations=20000, t_start=0.005, t_end=1e-6)
        
        pts_cart = bary_to_cart(best_bary)
        
        x0 = pts_cart.flatten()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            pts_polished = result.x.reshape(N, 2)
            score = min_area_vectorized(pts_polished)
            
            if score > best_score:
                best_score = score
                best_points_cart = pts_polished.copy()
                
        except Exception:
            score = sa_score
            if score > best_score:
                best_score = score
                best_points_cart = pts_cart.copy()
                
    if best_points_cart is not None:
        x = best_points_cart[:, 0]
        y = best_points_cart[:, 1]
        
        w = 2 * y / np.sqrt(3)
        u = 1 - 0.5 * w - x
        v = 1 - u - w
        
        u = np.maximum(u, 0)
        v = np.maximum(v, 0)
        w = np.maximum(w, 0)
        
        sums = u + v + w
        mask = sums > 0
        u[mask] /= sums[mask]
        v[mask] /= sums[mask]
        w[mask] /= sums[mask]
        
        best_points_cart = (u[:, np.newaxis] * A + v[:, np.newaxis] * B + w[:, np.newaxis] * C)
        
        final_score = min_area_vectorized(best_points_cart)
        if final_score > best_score:
            best_score = final_score
            
    return best_points_cart
