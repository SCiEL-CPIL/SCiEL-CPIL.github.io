import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    Normalized area = 0.5 * |det(bary_matrix)|
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to be valid (non-negative, sum to 1).
    This is a projection onto the simplex.
    """
    b = bary.copy()
    # Normalize first to handle scale
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    
    # Clip negative values to 0
    b = np.clip(b, 0, 1)
    
    # Re-normalize
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _objective_softmin(x_flat, beta=50.0):
    """
    Objective function for optimization using soft-min.
    x_flat: shape (N*3,)
    Returns negative of soft-min area (since we want to maximize, minimize negative).
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    
    r1 = bary[_TRIPLES[:, 0]]
    r2 = bary[_TRIPLES[:, 1]]
    r3 = bary[_TRIPLES[:, 2]]
    
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    areas = 0.5 * np.abs(dets)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - (1.0 / beta) * log_sum_exp
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _is_time_up():
        return time.time() > deadline - 2.0
    
    best_area = 0.0
    best_bary = None
    
    bounds = [(0.0, 1.0)] * (N * 3)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple restarts of L-BFGS-B with soft-min objective
    # L-BFGS-B is efficient for large-scale bound-constrained optimization.
    # We use a moderate beta for smoothness, allowing gradient-based methods to work well.
    
    n_restarts = 20
    
    for i in range(n_restarts):
        if _is_time_up():
            break
            
        # Generate initial guess
        # Use Dirichlet distribution for valid barycentric coords
        # Concentration parameter 1.0 gives uniform distribution on simplex
        x0 = rng.dirichlet([1.0, 1.0, 1.0], size=N).flatten()
        
        try:
            res = minimize(
                _objective_softmin,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if _is_time_up():
                break
                
            current_bary = _project_bary(res.x.reshape(N, 3))
            current_area = _min_area_bary(current_bary)
            
            if current_area > best_area:
                best_area = current_area
                best_bary = current_bary.copy()
                
        except Exception:
            pass
            
    # If we have time, refine the best solution with a higher beta (sharper min)
    if best_bary is not None and not _is_time_up():
        try:
            x0 = best_bary.flatten()
            res = minimize(
                lambda x: _objective_softmin(x, beta=200.0),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if not _is_time_up():
                refined_bary = _project_bary(res.x.reshape(N, 3))
                refined_area = _min_area_bary(refined_bary)
                
                if refined_area > best_area:
                    best_area = refined_area
                    best_bary = refined_bary.copy()
        except Exception:
            pass

    if best_bary is None:
        # Simple fallback: random points
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = rng.dirichlet([1, 1, 1], size=N)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)
