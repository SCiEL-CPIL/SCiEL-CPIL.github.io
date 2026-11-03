import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, SQRT3/2])

def _bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian coordinates.
    Point P = (1-u-v)*A + u*B + v*C. Since A=(0,0), P = u*B + v*C.
    u, v can be arrays of shape (N,).
    """
    x = u * B[0] + v * C[0]
    y = u * B[1] + v * C[1]
    return np.column_stack((x, y)) if np.ndim(u) > 0 else np.array([x, y])

def _project_to_bary_simplex(u, v):
    """Project u, v onto the valid barycentric simplex {u>=0, v>=0, u+v<=1}."""
    u = np.clip(u, 0.0, None)
    v = np.clip(v, 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return u, v

def _min_area_from_bary(u, v):
    """Compute the minimum area of any triangle formed by the points.
    u, v: arrays of shape (N,) representing barycentric coordinates.
    Returns the minimum area (not doubled).
    """
    points = _bary_to_cart(u, v)
    n = len(points)
    min_area = np.inf
    # Vectorized triple loop for speed
    # Precompute all pair differences? No, just loop. N=11 is small.
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                vec1 = pj - pi
                vec2 = pk - pi
                det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
                area = 0.5 * abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    params: array of shape (2*N,) representing u_i, v_i for each point.
    Returns negative of the minimum area.
    """
    u = params[0:N]
    v = params[N:2*N]
    u, v = _project_to_bary_simplex(u, v)
    min_area = _min_area_from_bary(u, v)
    return -min_area

def _structured_init_fps(rng, n_points=N):
    """
    Farthest Point Sampling initialization.
    Starts with vertices and midpoints, then adds points farthest from existing set.
    """
    # Start with a good spread: vertices and midpoints
    # Vertices in bary: (0,0), (1,0), (0,1)
    # Midpoints in bary: (0.5, 0), (0, 0.5), (0.5, 0.5)
    # Center: (1/3, 1/3)
    
    # Let's just use random points for FPS start, or specific good ones.
    # Better: Start with 3 vertices, then add midpoints, then FPS.
    u_list = [0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1/3]
    v_list = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1/3]
    
    # If n_points > 7, add more via FPS
    while len(u_list) < n_points:
        # Calculate distances from all points in simplex to existing points
        # Sample random points to find the farthest
        # Efficient FPS:
        # 1. Generate many random candidates
        # 2. For each candidate, find min dist to existing
        # 3. Pick the one with max min-dist
        
        # To be fast, we can just pick a random point far from others
        # Or use a grid.
        # Let's try a simple heuristic: random search for farthest
        best_dist = -1
        best_u, best_v = 0.5, 0.5
        
        # Generate 1000 candidates
        cand_u = rng.random(1000)
        cand_v = rng.random(1000)
        # Project to simplex
        cand_u, cand_v = _project_to_bary_simplex(cand_u, cand_v)
        
        # Convert to cart for distance
        cand_pts = _bary_to_cart(cand_u, cand_v)
        exist_pts = _bary_to_cart(np.array(u_list), np.array(v_list))
        
        # Min distance from each candidate to any existing point
        # dist[i, j] = ||cand_i - exist_j||
        # This might be memory heavy if n is large, but n=11 is fine.
        # 1000 * 11 distances.
        dists = np.linalg.norm(cand_pts[:, np.newaxis, :] - exist_pts[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        idx = np.argmax(min_dists)
        
        if min_dists[idx] > best_dist:
            best_dist = min_dists[idx]
            best_u = cand_u[idx]
            best_v = cand_v[idx]
            
        u_list.append(best_u)
        v_list.append(best_v)
        
    return np.array(u_list), np.array(v_list)

def _local_optimize_slsqp(u0, v0, deadline):
    """Optimize using SLSQP with bounds and constraints."""
    x0 = np.concatenate([u0, v0])
    
    # Bounds: 0 <= u <= 1, 0 <= v <= 1
    bounds = [(0, 1)] * N + [(0, 1)] * N
    
    # Constraints: u + v <= 1 for each point
    # SLSQP handles linear constraints well
    # We can define them as:
    # sum(u) - 1 <= 0 for each i
    # But SLSQP takes a single function for inequality constraints.
    # We can use a penalty in the objective or define multiple constraints.
    # scipy.optimize.minimize SLSQP supports:
    # constraints: [ {'type': 'ineq', 'fun': constraint_func} ]
    # constraint_func returns array of values >= 0.
    
    def constraints(params):
        u = params[0:N]
        v = params[N:2*N]
        # We want 1 - u - v >= 0
        return 1.0 - u - v
        
    res = minimize(_objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=[{'type': 'ineq', 'fun': constraints}],
                   options={'maxiter': 200, 'disp': False, 'ftol': 1e-12})
    
    return res

def _local_optimize_lbfgs(u0, v0, deadline):
    """Polish using L-BFGS-B. Note: L-BFGS-B doesn't handle general constraints well, 
    so we rely on the projection in the objective or just use it on a projected space?
    Actually, L-BFGS-B only handles bounds. So we should only use it if points are strictly inside.
    If points are on boundary, gradient is undefined.
    So L-BFGS-B is risky. We'll stick to SLSQP and maybe Nelder-Mead."""
    pass

def _local_optimize_nm(u0, v0, deadline):
    """Nelder-Mead as a backup."""
    x0 = np.concatenate([u0, v0])
    res = minimize(_objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'disp': False, 'xatol': 1e-10, 'fatol': 1e-12})
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Initializations
    inits = []
    
    # 1. FPS structured
    u_fps, v_fps = _structured_init_fps(rng)
    inits.append((u_fps, v_fps))
    
    # 2. Random restarts
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        u, v = _project_to_bary_simplex(u, v)
        inits.append((u, v))
        
    # 3. Grid-like
    # Try to place points in a regular pattern
    # 3x3 grid + 2 extra
    # This is hard to do perfectly in barycentric.
    # Let's just add a few more random ones.
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        u, v = _project_to_bary_simplex(u, v)
        inits.append((u, v))

    for u0, v0 in inits:
        if time.time() > deadline:
            break
            
        # Try SLSQP
        try:
            res = _local_optimize_slsqp(u0, v0, deadline)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
        # Try Nelder-Mead on the result of SLSQP (or initial if SLSQP failed)
        if time.time() > deadline:
            break
            
        try:
            x_start = best_params if best_params is not None else np.concatenate([u0, v0])
            # If we have a good best_params, we can try to refine it further?
            # No, NM is stochastic. Let's run NM on the initial u0, v0 as well.
            res_nm = _local_optimize_nm(u0, v0, deadline)
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback
        u, v = _project_to_bary_simplex(rng.random(N), rng.random(N))
        return _bary_to_cart(u, v)

    u = best_params[0:N]
    v = best_params[N:2*N]
    u, v = _project_to_bary_simplex(u, v)
    
    return _bary_to_cart(u, v)

# Test
if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Min Area:", _min_area_from_bary(pts[:,0], pts[:,1])) # Wait, pts are cart. Need to convert back or calculate directly.
    # For debugging, let's just print the points.
