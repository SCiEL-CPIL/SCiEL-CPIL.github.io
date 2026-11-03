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

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex.
    Simple normalization and clipping.
    """
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _objective_flat(x_flat):
    """
    Objective function for optimization.
    Returns negative of min area.
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    area = _min_area_bary(bary)
    return -area

def _generate_structured_inits():
    """
    Generate a list of diverse initial configurations based on symmetry and heuristics.
    """
    inits = []
    
    # 1. Grid-like distribution (3x3 grid + corners/center)
    # We can map a 2D grid to barycentric coords
    # Or just use random Dirichlet with different seeds for diversity
    rng = np.random.default_rng(42)
    
    # Random Dirichlet samples
    for seed in range(20):
        r = np.random.default_rng(seed)
        inits.append(r.dirichlet([1, 1, 1], size=N))
        
    # Symmetric configurations
    # Try to place points in a hexagonal lattice pattern
    # Center point
    # 6 points around center
    # 4 points further out?
    # This is complex to get right in barycentric coords manually, 
    # but we can try specific patterns.
    
    # Pattern 1: One in center, others distributed
    center = np.array([1/3, 1/3, 1/3])
    # 6 points at radius 0.3 from center in cartesian, then convert?
    # Easier: just use the random inits and let NM do the work.
    # The key is diversity.
    
    # Add a "corner-heavy" init
    corners = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ]
    # Fill rest randomly
    r = np.random.default_rng(99)
    rest = r.dirichlet([1, 1, 1], size=N - 3)
    init_corner = np.vstack([corners, rest])
    inits.append(init_corner)
    
    # Add a "center-heavy" init
    r = np.random.default_rng(100)
    init_center = r.dirichlet([1, 1, 1], size=N)
    # Pull them slightly towards center
    init_center = 0.5 * init_center + 0.5 * np.array([1/3, 1/3, 1/3])
    init_center = _project_bary(init_center)
    inits.append(init_center)

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _is_time_up():
        return time.time() > deadline - 2.0

    best_area = 0.0
    best_bary = None
    
    inits = _generate_structured_inits()
    
    # Shuffle inits to ensure we try diverse ones first if time runs out
    rng = np.random.default_rng(123)
    rng.shuffle(inits)
    
    for init in inits:
        if _is_time_up():
            break
            
        # Refine with Nelder-Mead
        # NM is robust for this size (33 variables)
        x0 = init.flatten()
        
        try:
            res_nm = minimize(
                _objective_flat,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
        except Exception:
            continue
            
        refined_bary = _project_bary(res_nm.x.reshape(N, 3))
        refined_area = _min_area_bary(refined_bary)
        
        if refined_area > best_area:
            best_area = refined_area
            best_bary = refined_bary.copy()
            
        # If we have time, do a second pass on the best so far with tighter tolerance?
        # Or just keep going with next init.
        
    if best_bary is None:
        # Fallback
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)
