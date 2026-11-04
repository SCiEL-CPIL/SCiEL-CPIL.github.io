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

def _params_to_bary(params):
    """Convert unconstrained params (x, y) to barycentric (u, v) inside the triangle.
    Uses softmax-like mapping to ensure u>0, v>0, u+v<1.
    """
    x = params[0:N]
    y = params[N:2*N]
    # To prevent overflow in exp, we can shift by max
    # But for N=11 and reasonable ranges, simple exp might be ok.
    # Let's use a stable version.
    # max_val = np.maximum(x, y)
    # ex = np.exp(x - max_val)
    # ey = np.exp(y - max_val)
    # e0 = np.exp(0 - max_val) # which is exp(-max_val)
    # denom = ex + ey + e0
    # u = ex / denom
    # v = ey / denom
    
    # Simpler: just clip x, y to [-20, 20] to avoid overflow/underflow issues
    x = np.clip(x, -20.0, 20.0)
    y = np.clip(y, -20.0, 20.0)
    
    ex = np.exp(x)
    ey = np.exp(y)
    denom = ex + ey + 1.0
    u = ex / denom
    v = ey / denom
    return u, v

def _bary_to_params(u, v):
    """Convert barycentric (u, v) to unconstrained params (x, y).
    Used for initialization.
    """
    # Clip to avoid log(0)
    eps = 1e-6
    u = np.clip(u, eps, 1.0 - 2*eps)
    v = np.clip(v, eps, 1.0 - 2*eps)
    # Ensure u+v <= 1-eps
    s = u + v
    mask = s > 1.0 - eps
    if np.any(mask):
        scale = (1.0 - eps) / s[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    x = np.log(u / (1.0 - u - v))
    y = np.log(v / (1.0 - u - v))
    return np.concatenate([x, y])

def _min_area_from_bary(u, v):
    """Compute the minimum area of any triangle formed by the points.
    u, v: arrays of shape (N,) representing barycentric coordinates.
    Returns the minimum area (not doubled).
    """
    points = _bary_to_cart(u, v)
    n = len(points)
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                vec1 = pj - pi
                vec2 = pk - pi
                det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
                area = 0.5 * abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(params):
    """
    params: array of shape (2*N,) representing x_i, y_i for each point.
    Returns negative of the minimum area.
    """
    u, v = _params_to_bary(params)
    min_area = _min_area_from_bary(u, v)
    return -min_area

def _structured_init_fps(rng, n_points=N):
    """
    Farthest Point Sampling initialization.
    Starts with vertices and midpoints, then adds points farthest from existing set.
    """
    u_list = [0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1/3]
    v_list = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1/3]
    
    while len(u_list) < n_points:
        best_dist = -1
        best_u, best_v = 0.5, 0.5
        
        cand_u = rng.random(1000)
        cand_v = rng.random(1000)
        # Project to simplex for distance calc
        cand_u, cand_v = np.clip(cand_u, 0, None), np.clip(cand_v, 0, None)
        s = cand_u + cand_v
        mask = s > 1.0
        if np.any(mask):
            cand_u[mask] /= s[mask]
            cand_v[mask] /= s[mask]
            
        cand_pts = _bary_to_cart(cand_u, cand_v)
        exist_pts = _bary_to_cart(np.array(u_list), np.array(v_list))
        
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Initializations in barycentric space
    inits_bary = []
    
    # 1. FPS structured
    u_fps, v_fps = _structured_init_fps(rng)
    inits_bary.append((u_fps, v_fps))
    
    # 2. Random restarts
    for _ in range(15):
        u = rng.random(N)
        v = rng.random(N)
        # Project to simplex
        u, v = np.clip(u, 0, None), np.clip(v, 0, None)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        inits_bary.append((u, v))

    # Convert inits to param space
    inits_params = []
    for u, v in inits_bary:
        try:
            p = _bary_to_params(u, v)
            inits_params.append(p)
        except Exception:
            pass

    for p0 in inits_params:
        if time.time() > deadline:
            break
            
        # Use Powell's method, which is derivative-free and robust for this kind of problem
        try:
            res = minimize(_objective, p0, method='Powell', 
                           options={'maxiter': 200, 'disp': False, 'ftol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: random points
        u, v = _structured_init_fps(rng)
        return _bary_to_cart(u, v)

    u, v = _params_to_bary(best_params)
    return _bary_to_cart(u, v)

# Test
if __name__ == "__main__":
    pts = construct()
    print(pts)
    # Calculate min area for debugging
    # Need to convert back to bary? No, just use cart points directly in a local function
    def min_area_cart(pts):
        n = len(pts)
        min_a = np.inf
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    a = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if a < min_a:
                        min_a = a
        return min_a
    print("Min Area:", min_area_cart(pts))
