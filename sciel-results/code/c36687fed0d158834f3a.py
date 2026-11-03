"""Heilbronn-11: Maximize minimum triangle area using Barycentric Annealing and Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Safety margin
DEADLINE = 140.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)

def _bary_to_cart(uv):
    """Convert (N, 2) barycentric (u, v) to (N, 2) Cartesian."""
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    # Vertices: A=(0,0), B=(1,0), C=(0.5, S3H)
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert (N, 2) Cartesian to (N, 2) barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _compute_areas(uv):
    """Compute areas of all triangles for given barycentric coords."""
    cart = _bary_to_cart(uv)
    p1 = cart[_T_I]
    p2 = cart[_T_J]
    p3 = cart[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _penalty(uv):
    """Penalty for barycentric coords outside [0,1]."""
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    return np.sum(np.maximum(0, -u)**2 + np.maximum(0, -v)**2 + np.maximum(0, -w)**2)

def _soft_min(areas, alpha):
    """Log-sum-exp soft minimum."""
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _bary_objective(uv_flat, alpha, penalty_weight):
    uv = uv_flat.reshape(N, 2)
    pen = _penalty(uv)
    if pen > 1e-3:
        return 1e6 + penalty_weight * pen
    areas = _compute_areas(uv)
    soft_min_val = _soft_min(areas, alpha)
    return -soft_min_val + penalty_weight * pen

def _symmetric_init():
    """
    D3 Symmetric Initialization:
    1 Center
    3 Near Vertices
    3 Near Edge Midpoints
    4 Inner Ring
    """
    uv = np.zeros((N, 2))
    
    # 1. Center (1/3, 1/3)
    uv[0] = [1/3, 1/3]
    
    # 2. Near Vertices (pushed in by factor 0.1)
    # Vertex A (1,0) -> (0.9, 0.1, 0.1) -> u=0.9, v=0.1
    # Vertex B (0,1) -> (0.1, 0.9, 0.1) -> u=0.1, v=0.9
    # Vertex C (0,0) -> (0.1, 0.1, 0.9) -> u=0.1, v=0.1
    eps = 0.1
    uv[1] = [1-eps, eps] # Near A
    uv[2] = [eps, 1-eps] # Near B
    uv[3] = [eps, eps]   # Near C
    
    # 3. Near Edge Midpoints
    # Mid AB (0.5, 0.5, 0) -> u=0.5, v=0.5
    # Mid BC (0.5, 0, 0.5) -> u=0.5, v=0.0
    # Mid CA (0, 0.5, 0.5) -> u=0.0, v=0.5
    # Push in towards center
    eps2 = 0.15
    # Mid AB: lerp((0.5,0.5), (1/3,1/3), eps2)
    uv[4] = [0.5*(1-eps2) + (1/3)*eps2, 0.5*(1-eps2) + (1/3)*eps2]
    # Mid BC: lerp((0.5,0.0), (1/3,1/3), eps2)
    uv[5] = [0.5*(1-eps2) + (1/3)*eps2, 0.0*(1-eps2) + (1/3)*eps2]
    # Mid CA: lerp((0.0,0.5), (1/3,1/3), eps2)
    uv[6] = [0.0*(1-eps2) + (1/3)*eps2, 0.5*(1-eps2) + (1/3)*eps2]
    
    # 4. Inner Ring of 4 points
    # Center is (1/3, 1/3). 
    # In Cartesian, center is (0.5, S3H/3).
    # We want 4 points around it.
    # Barycentric to Cartesian is linear.
    # Let's place them in Cartesian space relative to center, then convert back.
    # Or just perturb the barycentric center.
    # Simple approach: Place at specific barycentric offsets.
    # Since barycentric space is a triangle, "circle" is not trivial.
    # Let's use Cartesian for this part and convert.
    cart_center = _bary_to_cart(np.array([[1/3, 1/3]]))[0]
    r = 0.15
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        cart_pt = np.array([[cart_center[0] + dx, cart_center[1] + dy]])
        bary_pt = _cart_to_bary(cart_pt)
        uv[7+i] = bary_pt[0]
        
    return uv

def _random_bary():
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r[:, 0:2]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Symmetric Init
    uv_sym = _symmetric_init()
    candidates.append(uv_sym)
    
    # 2. Random Inits
    for _ in range(5):
        candidates.append(_random_bary())
        
    best_uv = None
    best_score = 0
    
    # Phase 1: Annealing
    alphas = [20, 100, 500, 2000]
    
    for alpha in alphas:
        if time.time() > deadline - 30:
            break
            
        for i, uv_init in enumerate(candidates):
            if time.time() > deadline - 30:
                break
                
            # Optimize
            try:
                res = minimize(
                    _bary_objective, 
                    uv_init.flatten(), 
                    method='Nelder-Mead',
                    args=(alpha, 100.0),
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                continue
                
            uv_opt = res.x.reshape(N, 2)
            
            # Check validity
            if _penalty(uv_opt) > 1e-6:
                # Project back if slightly out
                u = uv_opt[:, 0]
                v = uv_opt[:, 1]
                w = 1 - u - v
                # Simple clip
                u = np.maximum(0, u)
                v = np.maximum(0, v)
                w = np.maximum(0, w)
                s = u + v + w
                uv_opt = np.column_stack([u/s, v/s])
                
            areas = _compute_areas(uv_opt)
            min_area = np.min(areas)
            
            if min_area > best_score:
                best_score = min_area
                best_uv = uv_opt.copy()
                
            # Update candidate list: keep best, add this one
            candidates[i] = uv_opt
    
    # Phase 2: Active Set Polish
    if best_uv is not None and time.time() < deadline - 10:
        current_uv = best_uv.copy()
        current_score = _compute_areas(current_uv).min()
        
        for _ in range(10): # Max polish iterations
            if time.time() > deadline - 5:
                break
                
            areas = _compute_areas(current_uv)
            # Identify active set: triangles with area close to min
            threshold = current_score * 1.01 + 1e-6
            active_mask = areas <= threshold
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
            if len(active_indices) > 20:
                # Too many, pick the smallest 20
                smallest_idx = np.argsort(areas)[:20]
                active_indices = smallest_idx
                
            def active_obj(uv_flat):
                uv = uv_flat.reshape(N, 2)
                pen = _penalty(uv)
                if pen > 1e-3:
                    return 1e6 + 100 * pen
                areas = _compute_areas(uv)
                active_areas = areas[active_indices]
                # Maximize min of active areas
                min_a = np.min(active_areas)
                # Use a hard min surrogate for sharpness
                # Or just return -min_a
                # Nelder-Mead handles non-smooth
                return -min_a + 100 * pen
                
            try:
                res = minimize(
                    active_obj, 
                    current_uv.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
            except Exception:
                break
                
            new_uv = res.x.reshape(N, 2)
            if _penalty(new_uv) > 1e-6:
                continue
                
            new_areas = _compute_areas(new_uv)
            new_min = np.min(new_areas)
            
            if new_min > current_score + 1e-9:
                current_uv = new_uv
                current_score = new_min
            else:
                # If no improvement, break to save time
                break
                
        best_uv = current_uv
        best_score = current_score

    # Convert to Cartesian
    best_cart = _bary_to_cart(best_uv)
    
    # Final sanity check
    if not np.all(best_cart[:, 0] >= -1e-9) or not np.all(best_cart[:, 1] >= -1e-9):
         # Fallback to symmetric if something went wrong
         best_cart = _bary_to_cart(_symmetric_init())

    return best_cart

