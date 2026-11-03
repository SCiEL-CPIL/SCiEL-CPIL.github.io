"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11

# Vertices of the equilateral triangle with side length 1
# Area = sqrt(3)/4
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
TRIANGLE_AREA = np.sqrt(3) / 4

# Precompute triple indices
_idx_i = None
_idx_j = None
_idx_k = None

def _get_indices():
    global _idx_i, _idx_j, _idx_k
    if _idx_i is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _idx_i = i[mask]
        _idx_j = j[mask]
        _idx_k = k[mask]
    return _idx_i, _idx_j, _idx_k

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    cart = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    return cart

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    v0 = B - A
    v1 = C - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01
    
    v2 = cart - A
    d20 = v2 @ v0
    d21 = v2 @ v1
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    bary = np.column_stack([u, v, w])
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def _all_triangle_areas_cart(points):
    """Compute all triangle areas for N points (returns array of C(N,3) areas)."""
    ii, jj, kk = _get_indices()
    p1 = points[ii]
    p2 = points[jj]
    p3 = points[kk]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return areas

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    return np.min(_all_triangle_areas_cart(points))

def _softmin_objective(x, beta):
    """Differentiable soft-min surrogate of the minimum triangle area."""
    bary = np.array(x).reshape(N, 3)
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    points = _bary_to_cart(bary)
    areas = _all_triangle_areas_cart(points)
    # Soft minimum: -log(sum(exp(-areas/beta))) * beta
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(areas)
    log_sum = min_val / beta + np.log(np.sum(np.exp(-(areas - min_val) / beta)))
    return -beta * log_sum

def _hard_objective(x):
    """Non-differentiable hard minimum (for DE and final evaluation)."""
    bary = np.array(x).reshape(N, 3)
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    points = _bary_to_cart(bary)
    return -_min_triangle_area(points)

def _score_bary(bary):
    """Score a barycentric configuration (higher is better)."""
    points = _bary_to_cart(bary)
    return _min_triangle_area(points)

def _structured_init_bary():
    """Create a structured initial configuration in barycentric coordinates."""
    # Use a pattern that spreads points across the triangle
    bary = np.zeros((N, 3))
    # Place points in a rough grid pattern
    positions = [
        (0.1, 0.1, 0.8),
        (0.1, 0.8, 0.1),
        (0.8, 0.1, 0.1),
        (0.1, 0.4, 0.5),
        (0.5, 0.1, 0.4),
        (0.4, 0.5, 0.1),
        (0.3, 0.3, 0.4),
        (0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3),
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
    ]
    for i, (u, v, w) in enumerate(positions):
        bary[i] = [u, v, w]
    # Normalize
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def _random_init_bary(rng):
    """Generate random barycentric coordinates."""
    raw = rng.random((N, 3))
    bary = raw / raw.sum(axis=1, keepdims=True)
    return bary

def _l_bfgs_softmin(bary_init, beta):
    """Optimize using L-BFGS-B with soft-min objective."""
    x0 = bary_init.flatten()
    
    def obj(x):
        return _softmin_objective(x, beta)
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 3000, 'gtol': 1e-8})
        bary_opt = result.x.reshape(N, 3)
        bary_opt = np.clip(bary_opt, 0, None)
        bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
        return bary_opt
    except Exception:
        return bary_init

def _beta_anneal_refine(bary_init):
    """Refine with decreasing beta for sharper minimum approximation."""
    betas = [0.01, 0.005, 0.002, 0.001, 0.0005]
    bary = bary_init.copy()
    for beta in betas:
        bary = _l_bfgs_softmin(bary, beta)
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (5s before 150s limit)
    
    best_bary = None
    best_value = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        bounds = [(0, 1)] * (N * 3)
        
        def de_objective(x):
            return _hard_objective(x)
        
        try:
            de_result = differential_evolution(
                de_objective, bounds, maxiter=50, popsize=20,
                tol=1e-8, seed=42, polish=True, 
                workers=1, init='sobol'
            )
            de_bary = de_result.x.reshape(N, 3)
            de_bary = np.clip(de_bary, 0, None)
            de_bary = de_bary / de_bary.sum(axis=1, keepdims=True)
            de_value = _score_bary(de_bary)
            
            if de_value > best_value:
                best_value = de_value
                best_bary = de_bary
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement with soft-min
    n_starts = 15
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            bary_init = _structured_init_bary()
        elif i == 1:
            bary_init = _random_init_bary(rng)
        elif i == 2:
            bary_init = _random_init_bary(rng)
        else:
            if best_bary is not None:
                # Perturb best
                bary_init = best_bary + rng.normal(0, 0.02, (N, 3))
                bary_init = np.clip(bary_init, 0, None)
                bary_init = bary_init / bary_init.sum(axis=1, keepdims=True)
            else:
                bary_init = _random_init_bary(rng)
        
        # Refine with beta annealing
        bary_refined = _beta_anneal_refine(bary_init)
        value = _score_bary(bary_refined)
        
        if value > best_value:
            best_value = value
            best_bary = bary_refined
    
    # Phase 3: Final polish with very high beta
    if best_bary is not None and time.time() < deadline - 5:
        best_bary = _l_bfgs_softmin(best_bary, 0.0001)
        final_value = _score_bary(best_bary)
        if final_value > best_value:
            best_value = final_value
    
    if best_bary is None:
        # Fallback
        best_bary = _random_init_bary(rng)
    
    # Convert to Cartesian and return
    best_points = _bary_to_cart(best_bary)
    return best_points
