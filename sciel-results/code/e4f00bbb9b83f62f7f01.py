import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Triangle vertices
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, _H])

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute pair indices for distance objective
_pair_combos = list(combinations(range(N), 2))
_pidx_i = np.array([c[0] for c in _pair_combos])
_pidx_j = np.array([c[1] for c in _pair_combos])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N,3) to Cartesian (N,2)."""
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2


def _cart_to_bary(pts):
    """Convert Cartesian (N,2) to barycentric coordinates (N,3)."""
    # barycentric: point = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # x = b1 + 0.5*b2, y = H*b2
    # b2 = y/H, b1 = x - 0.5*b2, b0 = 1 - b1 - b2
    b2 = pts[:, 1] / _H
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _min_triangle_area_cartesian(points):
    """Compute minimum area of all C(11,3) triangles from Cartesian points."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _soft_min_bary(bary_flat, beta):
    """
    Compute soft minimum of triangle areas in barycentric coordinates.
    Returns negative soft-min area (to be minimized).
    """
    bary = bary_flat.reshape(N, 3)
    pts = _bary_to_cart(bary)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    # Soft minimum: -log(sum(exp(-area/beta))) * beta
    shifted = -areas / beta
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = -beta * (log_sum + np.max(shifted))
    return soft_min


def _objective_bary(bary_flat, beta):
    """Objective function for barycentric optimization: minimize -soft_min_area."""
    return -_soft_min_bary(bary_flat, beta)


def _trf_optimize_bary(bary_init, beta, time_budget=5.0):
    """
    Trust-region reflective optimization in barycentric coordinates.
    Constraints: each barycentric coordinate in [0,1], and they sum to 1.
    We parameterize with 2 free coordinates per point (b1, b2), b0 = 1-b1-b2.
    """
    start_time = time.time()
    
    # Extract free parameters: b1, b2 for each point (22 dims)
    # b0 = 1 - b1 - b2
    def extract_free(bary):
        return bary[:, 1:3].flatten()
    
    def reconstruct(free):
        b1 = free[0::2]
        b2 = free[1::2]
        b0 = 1.0 - b1 - b2
        return np.column_stack([b0, b1, b2])
    
    free_init = extract_free(bary_init)
    
    # Bounds: b1 >= 0, b2 >= 0, b0 >= 0 => b1+b2 <= 1
    # For trf, we need box constraints. Use b1 in [0,1], b2 in [0, 1-b1]
    # But trf only supports box constraints directly.
    # Alternative: use slack variable or just box [0,1] and project.
    # Let's use box [0, 1] for both and add penalty for b1+b2 > 1.
    
    lb = np.zeros(2 * N)
    ub = np.ones(2 * N)
    
    def obj(free):
        if time.time() - start_time > time_budget:
            return 1e10
        bary = reconstruct(free)
        # Penalty for b1+b2 > 1 (b0 < 0)
        b0 = 1.0 - free[0::2] - free[1::2]
        penalty = 100.0 * np.sum(np.maximum(0, -b0)**2)
        return _objective_bary(bary.flatten(), beta) + penalty
    
    try:
        res = minimize(obj, free_init, method='trf', bounds=(lb, ub),
                       options={'maxiter': 500, 'xatol': 1e-10, 'gtol': 1e-10,
                                'ftol': 1e-10, 'max_nfev': 2000})
        free_opt = res.x
        # Project to valid barycentric
        b1 = np.clip(free_opt[0::2], 0, 1)
        b2 = np.clip(free_opt[1::2], 0, 1)
        b0 = 1.0 - b1 - b2
        # If b0 < 0, scale b1 and b2
        mask = b0 < 0
        if np.any(mask):
            s = 1.0 / (b1[mask] + b2[mask])
            b1[mask] *= s
            b2[mask] *= s
            b0[mask] = 0.0
        bary_opt = np.column_stack([b0, b1, b2])
        return bary_opt
    except Exception:
        return bary_init


def _lbfgs_optimize_bary(bary_init, beta, time_budget=5.0):
    """
    L-BFGS-B optimization in barycentric coordinates with penalty.
    """
    start_time = time.time()
    
    def extract_free(bary):
        return bary[:, 1:3].flatten()
    
    def reconstruct(free):
        b1 = free[0::2]
        b2 = free[1::2]
        b0 = 1.0 - b1 - b2
        return np.column_stack([b0, b1, b2])
    
    free_init = extract_free(bary_init)
    
    lb = np.zeros(2 * N)
    ub = np.ones(2 * N)
    
    def obj(free):
        if time.time() - start_time > time_budget:
            return 1e10
        bary = reconstruct(free)
        b0 = 1.0 - free[0::2] - free[1::2]
        penalty = 100.0 * np.sum(np.maximum(0, -b0)**2)
        return _objective_bary(bary.flatten(), beta) + penalty
    
    try:
        res = minimize(obj, free_init, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                       options={'maxiter': 300, 'xatol': 1e-10, 'ftol': 1e-10,
                                'maxfun': 1000})
        free_opt = res.x
        b1 = np.clip(free_opt[0::2], 0, 1)
        b2 = np.clip(free_opt[1::2], 0, 1)
        b0 = 1.0 - b1 - b2
        mask = b0 < 0
        if np.any(mask):
            s = 1.0 / (b1[mask] + b2[mask])
            b1[mask] *= s
            b2[mask] *= s
            b0[mask] = 0.0
        bary_opt = np.column_stack([b0, b1, b2])
        return bary_opt
    except Exception:
        return bary_init


def _polish_bary(bary_init, time_budget=3.0):
    """
    Polish using SLSQP on the true (non-smooth) minimum area objective.
    """
    start_time = time.time()
    
    def extract_free(bary):
        return bary[:, 1:3].flatten()
    
    def reconstruct(free):
        b1 = free[0::2]
        b2 = free[1::2]
        b0 = 1.0 - b1 - b2
        return np.column_stack([b0, b1, b2])
    
    free_init = extract_free(bary_init)
    
    lb = np.zeros(2 * N)
    ub = np.ones(2 * N)
    
    def obj(free):
        if time.time() - start_time > time_budget:
            return 1e10
        bary = reconstruct(free)
        pts = _bary_to_cart(bary)
        return -_min_triangle_area_cartesian(pts)
    
    try:
        res = minimize(obj, free_init, method='SLSQP', bounds=list(zip(lb, ub)),
                       options={'maxiter': 100, 'ftol': 1e-12, 'eps': 1e-8})
        free_opt = res.x
        b1 = np.clip(free_opt[0::2], 0, 1)
        b2 = np.clip(free_opt[1::2], 0, 1)
        b0 = 1.0 - b1 - b2
        mask = b0 < 0
        if np.any(mask):
            s = 1.0 / (b1[mask] + b2[mask])
            b1[mask] *= s
            b2[mask] *= s
            b0[mask] = 0.0
        bary_opt = np.column_stack([b0, b1, b2])
        return bary_opt
    except Exception:
        return bary_init


def _generate_symmetric_bary():
    """Generate symmetric initialization in barycentric coordinates."""
    points_cart = []
    
    # Vertices
    points_cart.append([0.0, 0.0])
    points_cart.append([1.0, 0.0])
    points_cart.append([0.5, _H])
    
    # Centroid
    points_cart.append([0.5, _H / 3])
    
    # Midpoints of edges
    points_cart.append([0.5, 0.0])
    points_cart.append([0.25, _H / 2])
    points_cart.append([0.75, _H / 2])
    
    # Points at 1/3 and 2/3 along medians
    points_cart.append([0.5 / 3, _H / 3])
    points_cart.append([0.5 * 2 / 3, _H * 2 / 3])
    
    points_cart.append([1.0 - 0.5 / 3, _H / 3])
    points_cart.append([1.0 - 0.5 * 2 / 3, _H * 2 / 3])
    
    pts = np.array(points_cart[:N])
    return _cart_to_bary(pts)


def _generate_random_bary(rng, n=N):
    """Generate random barycentric coordinates."""
    bary = np.zeros((n, 3))
    for i in range(n):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        bary[i] = [1 - r1 - r2, r1, r2]
    return bary


def _generate_structured_bary():
    """Generate a structured grid-like initialization in barycentric coords."""
    points_cart = []
    
    # Grid-like pattern: rows of points
    # Row 0 (bottom): 4 points
    for i in range(4):
        t = i / 3.0
        points_cart.append([t, 0.0])
    
    # Row 1 (middle): 3 points
    for i in range(3):
        t = (i + 0.5) / 3.0
        y = _H / 3
        x = t
        points_cart.append([x, y])
    
    # Row 2 (top): 2 points
    for i in range(2):
        t = (i + 0.5) / 2.0
        y = 2 * _H / 3
        x = 0.5 + (t - 0.5) * 0.5
        points_cart.append([x, y])
    
    # Add centroid
    points_cart.append([0.5, _H / 3])
    
    pts = np.array(points_cart[:N])
    return _cart_to_bary(pts)


def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle.
    Uses barycentric coordinate clipping.
    """
    bary = _cart_to_bary(pts)
    # Clip to [0, 1] and renormalize
    bary = np.clip(bary, 0, None)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary = bary / row_sums
    return _bary_to_cart(bary)


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    return _min_triangle_area_cartesian(points)


def _min_dist_sq(points):
    """Compute minimum squared distance between any pair."""
    pi = points[_pidx_i]
    pj = points[_pidx_j]
    diff = pi - pj
    dists = np.sum(diff * diff, axis=1)
    return np.min(dists)


def _local_optimize_bary(bary_init, time_budget=10.0):
    """
    Multi-stage optimization in barycentric coordinates:
    1. L-BFGS-B with soft-min surrogate (beta annealing)
    2. TRF with tighter beta
    3. SLSQP polish on true objective
    """
    start_time = time.time()
    
    bary = bary_init.copy()
    
    # Stage 1: L-BFGS-B with soft-min, anneal beta
    betas = [200.0, 100.0, 50.0, 20.0, 10.0]
    for beta in betas:
        if time.time() - start_time > time_budget * 0.6:
            break
        remaining = time_budget * 0.6 - (time.time() - start_time)
        if remaining < 0.5:
            break
        bary = _lbfgs_optimize_bary(bary, beta, time_budget=min(3.0, remaining))
    
    # Stage 2: TRF with tighter beta
    if time.time() - start_time < time_budget * 0.8:
        remaining = time_budget * 0.8 - (time.time() - start_time)
        if remaining > 0.5:
            bary = _trf_optimize_bary(bary, 5.0, time_budget=min(2.0, remaining))
    
    # Stage 3: SLSQP polish on true objective
    if time.time() - start_time < time_budget * 0.95:
        remaining = time_budget * 0.95 - (time.time() - start_time)
        if remaining > 0.5:
            bary = _polish_bary(bary, time_budget=min(3.0, remaining))
    
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initialization candidates in barycentric coordinates
    init_candidates = []
    
    # 1. Symmetric initialization
    sym_bary = _generate_symmetric_bary()
    init_candidates.append(sym_bary)
    
    # 2. Structured grid initialization
    struct_bary = _generate_structured_bary()
    init_candidates.append(struct_bary)
    
    # 3. Random initializations
    for i in range(6):
        rand_bary = _generate_random_bary(rng)
        init_candidates.append(rand_bary)
    
    # Process each candidate with local optimization
    n_candidates = len(init_candidates)
    
    for idx, init_bary in enumerate(init_candidates):
        if time.time() > deadline - 5:
            break
        
        try:
            # Allocate time based on remaining budget
            remaining = deadline - time.time() - 3
            time_per_candidate = max(3.0, remaining / max(1, n_candidates - idx))
            
            optimized_bary = _local_optimize_bary(init_bary, time_budget=time_per_candidate)
            pts = _bary_to_cart(optimized_bary)
            score = _min_triangle_area_cartesian(pts)
            
            if score > best_score:
                best_score = score
                best_bary = optimized_bary.copy()
                
        except Exception:
            continue
    
    # If we have time, do additional rounds of perturbation + optimization
    if best_bary is not None and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution in barycentric space
            perturbed = best_bary.copy()
            i = rng.integers(0, N)
            scale = 0.03
            perturbed[i] += rng.normal(0, scale, size=3)
            # Renormalize
            perturbed[i] = np.clip(perturbed[i], 0, None)
            s = perturbed[i].sum()
            if s > 1e-10:
                perturbed[i] /= s
            else:
                perturbed[i] = best_bary[i]
            
            # Optimize
            remaining = deadline - time.time() - 3
            time_budget = max(3.0, remaining / 3)
            optimized = _local_optimize_bary(perturbed, time_budget=time_budget)
            pts = _bary_to_cart(optimized)
            score = _min_triangle_area_cartesian(pts)
            
            if score > best_score:
                best_score = score
                best_bary = optimized.copy()
    
    if best_bary is None:
        # Fallback: symmetric initialization
        best_bary = _generate_symmetric_bary()
    
    return _bary_to_cart(best_bary)

# EVOLVE-BLOCK-END
