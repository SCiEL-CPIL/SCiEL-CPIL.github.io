import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
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
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Cross product r2 x r3
    cross_r2_r3 = np.cross(r2, r3)
    
    # Dot product r1 . (r2 x r3)
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
    Simple clip and normalize.
    """
    b = bary.copy()
    b = np.clip(b, 1e-8, 1.0)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b = b / b_sum
    return b

def _objective_softmin(x_flat, beta=100.0):
    """
    Objective function: negative of the softmin of the areas.
    softmin(A) = -1/beta * log(sum(exp(-beta * A)))
    This is a smooth approximation of min(A).
    As beta increases, it converges to min(A).
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    
    r1 = bary[_TRIPLES[:, 0]]
    r2 = bary[_TRIPLES[:, 1]]
    r3 = bary[_TRIPLES[:, 2]]
    
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    areas = 0.5 * np.abs(dets)
    
    # Handle numerical stability for log-sum-exp
    # We want min(areas), so we maximize -min(areas).
    # softmin = -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, subtract max exponent (which is -beta * min_area)
    # Let z = -beta * areas
    # max_z = -beta * min(areas)
    # sum(exp(z)) = exp(max_z) * sum(exp(z - max_z))
    # log(sum(exp(z))) = max_z + log(sum(exp(z - max_z)))
    # softmin = -1/beta * (max_z + log_sum)
    
    min_area = np.min(areas)
    z = -beta * areas
    max_z = np.max(z) # This is -beta * min_area
    
    # If areas are all 0, this might be problematic, but points are distinct usually.
    # exp(z - max_z) is stable.
    sum_exp = np.sum(np.exp(z - max_z))
    
    # log_sum_exp = max_z + log(sum_exp)
    # softmin = -1/beta * log_sum_exp
    
    # We want to minimize -softmin
    # -softmin = 1/beta * log_sum_exp
    return (1.0 / beta) * (max_z + np.log(sum_exp))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _is_time_up():
        return time.time() > deadline - 2.0
        
    best_area = 0.0
    best_bary = None
    
    # Multi-start strategy with L-BFGS-B
    # We use different initial configurations and beta values.
    
    # Initial guesses:
    # 1. Random uniform in barycentric coords
    # 2. Grid-like structure
    # 3. Center + perimeter
    
    rng = np.random.default_rng(42)
    
    # Start with a moderate beta to allow smooth gradients, then increase?
    # Or just use a high beta with a good start?
    # Let's try a few runs with different betas and starts.
    
    runs = []
    
    # Run 1: Random start, moderate beta
    bary0 = rng.dirichlet([1, 1, 1], size=N)
    runs.append((bary0, 50.0))
    
    # Run 2: Random start, high beta
    bary1 = rng.dirichlet([1, 1, 1], size=N)
    runs.append((bary1, 100.0))
    
    # Run 3: Structured start (approximate grid)
    # Try to place points roughly on a grid in barycentric space
    # 3x3 grid + center? 11 points.
    # Barycentric grid points for 3x3: (i/2, j/2, k/2) where i+j+k=2? No, sum must be 1.
    # Let's just use a few heuristic starts.
    
    # Heuristic: 1 center, 10 on perimeter?
    # Or just random is often best for 11 points.
    
    # Run 4: Another random start
    bary2 = rng.dirichlet([1, 1, 1], size=N)
    runs.append((bary2, 200.0))
    
    # Run 5: Another random start
    bary3 = rng.dirichlet([1, 1, 1], size=N)
    runs.append((bary3, 500.0))
    
    for bary_init, beta in runs:
        if _is_time_up():
            break
            
        x0 = bary_init.flatten()
        
        # Use L-BFGS-B
        # Bounds: barycentric coords must be in [0, 1]
        # But we project inside the objective, so bounds are less critical for validity,
        # but help the optimizer stay in a reasonable region.
        bounds = [(0.0, 1.0)] * (N * 3)
        
        try:
            res = minimize(
                _objective_softmin,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if _is_time_up():
                break
                
            current_bary = _project_bary(res.x.reshape(N, 3))
            current_area = _min_area_bary(current_bary)
            
            if current_area > best_area:
                best_area = current_area
                best_bary = current_bary.copy()
                
        except Exception:
            continue
            
    # If we have time, refine the best result with Nelder-Mead on the exact min objective
    # This helps because softmin is an approximation.
    if best_bary is not None and not _is_time_up():
        x0 = best_bary.flatten()
        try:
            res_nm = minimize(
                lambda x: -_min_area_bary(_project_bary(x.reshape(N, 3))),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if not _is_time_up():
                refined_bary = _project_bary(res_nm.x.reshape(N, 3))
                refined_area = _min_area_bary(refined_bary)
                if refined_area > best_area:
                    best_area = refined_area
                    best_bary = refined_bary.copy()
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)

    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = rng.dirichlet([1, 1, 1], size=N)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)
