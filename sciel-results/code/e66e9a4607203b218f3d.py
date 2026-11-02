"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit. 
# The evaluator kills after 150s. We aim to return well before.
DEADLINE_MARGIN = 145.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])

def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2
    """
    uv = uv_flat.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # P = u*V0 + v*V1 + w*V2
    # V0 = (0,0), V1 = (1,0), V2 = (0.5, S3H)
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """
    Convert Cartesian (N, 2) to barycentric coordinates (N, 2) (u, v).
    x = v + 0.5w
    y = S3H * w  => w = y / S3H = 2y/S3
    v = x - 0.5w
    u = 1 - v - w
    """
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _vectorized_areas_cart(points):
    """Calculate all triangle areas formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    areas = _vectorized_areas_cart(points)
    return np.min(areas)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    areas = _vectorized_areas_cart(points)
    # Soft min: -1/alpha * log(sum(exp(-alpha * areas)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    # If alpha is very large, this approaches the hard min
    # We want to MAXIMIZE the soft min, so the objective function will be -soft_min
    exp_terms = np.exp(-alpha * (areas - max_a))
    log_sum_exp = np.log(np.sum(exp_terms))
    soft_min_val = max_a - (1.0 / alpha) * log_sum_exp
    return soft_min_val

def _project_to_triangle_bary(uv):
    """
    Project barycentric coordinates (u, v) to be valid (u,v >= 0, u+v <= 1).
    This is a projection onto the standard 2-simplex.
    """
    u = uv[:, 0].copy()
    v = uv[:, 1].copy()
    
    # Simple clipping is not a projection, but for Nelder-Mead it often works well enough
    # if we penalize outside regions heavily. However, a better approach is to clip
    # and renormalize if sum > 1, or just clip to [0,1] and ensure u+v<=1.
    
    # Clip to 0
    u = np.maximum(0, u)
    v = np.maximum(0, v)
    
    # If u + v > 1, scale down
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
        
    return np.column_stack([u, v])

def _bary_objective(flat_uv, alpha=100):
    """
    Objective function in barycentric coordinates.
    Minimizes -soft_min_area + penalty.
    """
    uv = flat_uv.reshape(N, 2)
    
    # Project to valid barycentric region
    uv = _project_to_triangle_bary(uv)
    
    cart = _bary_to_cart(uv.flatten())
    
    # Calculate soft min area
    soft_min_val = _soft_min_area_cart(cart, alpha)
    
    # Penalty for being outside the triangle (should be 0 after projection, 
    # but adds robustness if projection logic has edge cases)
    x = cart[:, 0]
    y = cart[:, 1]
    # y >= 0
    pen1 = np.sum(np.maximum(0, -y)**2)
    # y <= S3 * x
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    # S3 * x + y <= S3
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    
    penalty = 100.0 * (pen1 + pen2 + pen3)
    
    return -soft_min_val + penalty

def _cart_objective(flat_cart):
    """
    Objective function in Cartesian coordinates for final polish.
    Minimizes -min_area + penalty.
    """
    cart = flat_cart.reshape(N, 2)
    
    # Calculate hard min area
    min_val = _min_area_cart(cart)
    
    # Penalty for being outside the triangle
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    
    penalty = 100.0 * (pen1 + pen2 + pen3)
    
    return -min_val + penalty

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    best_uv = None
    best_score = -float('inf')
    best_cart = None
    
    # Annealing schedule: start with low alpha (loose soft-min) to explore,
    # then increase alpha to sharpen the minimum.
    alphas = [10, 50, 100, 200, 500, 1000]
    
    # Initial best: random
    uv_init = np.random.random((N, 2))
    uv_init = _project_to_triangle_bary(uv_init)
    cart_init = _bary_to_cart(uv_init.flatten())
    score_init = _min_area_cart(cart_init)
    
    if score_init > best_score:
        best_score = score_init
        best_uv = uv_init.copy()
        best_cart = cart_init.copy()
        
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        # Multiple restarts at each alpha level
        # As alpha increases, we rely more on the previous best
        num_restarts = 5 if alpha < 100 else 3
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_uv.flatten()
            else:
                # Perturb the best solution or random
                if np.random.random() < 0.5:
                    x0 = best_uv.flatten() + np.random.normal(0, 0.1, N*2)
                else:
                    x0 = np.random.random(N*2)
                
                x0 = _project_to_triangle_bary(x0.reshape(N, 2)).flatten()
            
            # Optimize with Nelder-Mead in barycentric space
            try:
                res = minimize(_bary_objective, x0, method='Nelder-Mead',
                              args=(alpha,),
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                new_uv = res.x.reshape(N, 2)
                new_uv = _project_to_triangle_bary(new_uv)
                new_cart = _bary_to_cart(new_uv.flatten())
                
                new_score = _min_area_cart(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_uv = new_uv
                    best_cart = new_cart
            except Exception:
                pass
    
    # Final refinement with hard minimum in Cartesian space
    # This helps to precisely maximize the single smallest triangle
    if time.time() < deadline - 5:
        x0 = best_cart.flatten()
        try:
            res = minimize(_cart_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            new_cart = res.x.reshape(N, 2)
            new_score = _min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart
                # Update barycentric for consistency, though we return Cartesian
                best_uv = _cart_to_bary(best_cart)
        except Exception:
            pass
            
    # Ensure final result is valid
    if best_cart is None:
        # Fallback
        best_cart = np.array([[0.5, S3H/3]] * N)
        
    # Final check and clamp
    x = best_cart[:, 0]
    y = best_cart[:, 1]
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    best_cart = np.column_stack([x, y])
    
    return best_cart
# EVOLVE-BLOCK-END
