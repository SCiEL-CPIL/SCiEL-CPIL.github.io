"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Vertices of the unit triangle in Cartesian
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, H])


def _bary_to_cart(bary):
    """
    Convert barycentric coordinates to Cartesian.
    Input: (N, 2) array of (u, v) where w = 1 - u - v.
    Output: (N, 2) array of (x, y).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * _V1[0] + v * _V2[0]
    y = u * _V1[1] + v * _V2[1]
    return np.column_stack([x, y])


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _objective_bary(flat):
    """
    Objective function for barycentric optimization.
    Minimizes negative min triangle area.
    Input: flat array of length 2*N (20 values).
    """
    # Reshape to (N, 2)
    uv = flat.reshape(N, 2)
    
    # Clip to ensure valid barycentric coordinates (u>=0, v>=0, u+v<=1)
    # This acts as a soft constraint or we can just let the optimizer handle it
    # but clipping ensures we stay in the domain for area calculation.
    # However, for optimization, it's better to let the optimizer see the boundaries.
    # We will use bounds in the optimizer call.
    
    # Convert to Cartesian
    points = _bary_to_cart(uv)
    
    # Calculate min area
    min_area = _min_triangle_area_fast(points)
    
    # We want to maximize min_area, so we minimize -min_area
    return -min_area


def _generate_initials():
    """Generate diverse initial configurations in barycentric space."""
    inits = []
    
    # 1. Symmetric corners + center
    base_bary = np.array([
        [0.05, 0.05], # Near V0 (0,0) -> w=0.9
        [0.95, 0.05], # Near V1 (1,0) -> w=0
        [0.05, 0.95], # Near V2 (0.5, H) -> w=0
        [0.5, 0.0],   # Mid V0-V1
        [0.0, 0.5],   # Mid V0-V2
        [0.5, 0.5],   # Mid V1-V2 (w=0)
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.4, 0.4]
    ])
    inits.append(base_bary)
    
    # 2. Grid-like
    grid_bary = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
        [0.3, 0.5], [0.5, 0.5],
        [0.4, 0.7]
    ])
    inits.append(grid_bary)
    
    # 3. Random
    rng = np.random.RandomState(42)
    for _ in range(10):
        # Sample from Dirichlet(1,1,1)
        bary = rng.dirichlet([1, 1, 1], size=N)
        # We store (u, v)
        inits.append(np.column_stack([bary[:, 0], bary[:, 1]]))
        
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_points_cart = None
    best_score = -np.inf
    
    # Define bounds for u, v
    # u >= 0, v >= 0, u + v <= 1
    # We can't easily enforce u+v<=1 with simple bounds in L-BFGS-B/Powell without penalty.
    # However, Powell handles box constraints.
    # To handle u+v<=1, we can use a transformation or penalty.
    # Let's use a penalty in the objective if we go out of bounds, or clip.
    # Clipping in objective is non-differentiable but Powell is derivative-free.
    # Let's stick to box constraints [0, 1] for u and v, and add a penalty for u+v > 1.
    
    def _objective_with_penalty(flat):
        uv = flat.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        
        # Penalty for u+v > 1
        viol = np.maximum(0, u + v - 1)
        penalty = np.sum(viol) * 1000
        
        # Also ensure u,v >= 0 (handled by bounds)
        
        points = _bary_to_cart(uv)
        min_area = _min_triangle_area_fast(points)
        
        return -min_area + penalty

    bounds = [(0, 1)] * (2 * N)

    inits = _generate_initials()
    
    # Phase 1: Local refinement with Powell from multiple starts
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        try:
            flat_init = init.flatten()
            
            # Powell is robust for non-smooth functions
            res = minimize(
                _objective_with_penalty,
                flat_init,
                method='Powell',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'xrtol': 1e-10,
                    'fatol': 1e-12,
                    'maxfev': 10000
                }
            )
            
            # Check if result is valid (u+v <= 1 + eps)
            uv_res = res.x.reshape(N, 2)
            if np.all(uv_res[:, 0] + uv_res[:, 1] <= 1.0 + 1e-8):
                points_res = _bary_to_cart(uv_res)
                score_res = _min_triangle_area_fast(points_res)
                
                if score_res > best_score:
                    best_score = score_res
                    best_points_cart = points_res.copy()
                    
        except Exception:
            continue
            
    # Phase 2: If we have a good candidate, try to refine it further with Nelder-Mead
    # Nelder-Mead can sometimes polish better than Powell in the final stages
    if best_points_cart is not None:
        if time.time() < deadline - 5:
            try:
                # Convert best Cartesian back to Barycentric for refinement?
                # Or just refine in Cartesian?
                # Cartesian refinement requires clipping.
                # Let's stick to Barycentric for consistency.
                # We need to convert best_points_cart to barycentric.
                # x = u + 0.5v, y = H*v => v = y/H, u = x - 0.5v
                v = best_points_cart[:, 1] / H
                u = best_points_cart[:, 0] - 0.5 * v
                # Clip u,v to [0,1] and u+v<=1
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                # Ensure u+v <= 1
                scale = np.minimum(1.0, 1.0 / np.maximum(u + v, 1e-9))
                u = u * scale
                v = v * scale
                
                flat_best = np.column_stack([u, v]).flatten()
                
                res_final = minimize(
                    _objective_with_penalty,
                    flat_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                
                uv_final = res_final.x.reshape(N, 2)
                if np.all(uv_final[:, 0] + uv_final[:, 1] <= 1.0 + 1e-8):
                    points_final = _bary_to_cart(uv_final)
                    score_final = _min_triangle_area_fast(points_final)
                    
                    if score_final > best_score:
                        best_score = score_final
                        best_points_cart = points_final.copy()
                        
            except Exception:
                pass

    # Fallback
    if best_points_cart is None or not np.all(np.isfinite(best_points_cart)):
        # Use the first init as fallback
        fallback_bary = inits[0]
        best_points_cart = _bary_to_cart(fallback_bary)
        
    return best_points_cart
# EVOLVE-BLOCK-END
