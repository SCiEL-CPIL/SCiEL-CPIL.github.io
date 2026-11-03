import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = np.sqrt(3)

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, SQRT3 / 2])
TRI_AREA = SQRT3 / 4.0

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLES)

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) for vertices B and C (A is origin)
    to Cartesian coordinates.
    P = u * B + v * C
    """
    x = bary[:, 0] * B[0] + bary[:, 1] * C[0]
    y = bary[:, 0] * B[1] + bary[:, 1] * C[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """
    Project barycentric coordinates (u, v) to the valid region:
    u >= 0, v >= 0, u + v <= 1
    """
    b = bary.copy()
    b = np.clip(b, 0.0, None)
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0
    if np.any(mask):
        b[mask] = b[mask] / s[mask, None]
    return b

def _min_area(points):
    """
    points: array of shape (N, 2) Cartesian coordinates.
    Returns the minimum area of any triangle formed by 3 points.
    """
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_areas(points):
    """
    points: array of shape (N, 2) Cartesian coordinates.
    Returns all triangle areas.
    """
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _softmin_objective(x, beta=100.0):
    """
    Objective function: negative of the softmin of the areas.
    x: array of shape (N, 2) with barycentric coords (u, v)
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # softmin = -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area - log_sum / beta
    return -soft_min

def _hard_objective(x):
    """
    Objective function: negative of the min area.
    x: array of shape (N, 2) with barycentric coords (u, v)
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_area(points)

def _score_bary(bary):
    """
    Score a barycentric configuration.
    """
    points = _bary_to_cart(bary)
    return _min_area(points)

def _generate_initial_bary(seed, mode=0):
    """
    Generate diverse initial configurations in barycentric coords (u, v).
    """
    rng = np.random.default_rng(seed)
    
    if mode == 0:
        # Symmetric pattern: vertices, edge midpoints, centroid, plus some interior
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.2, 0.1], [0.7, 0.1], [0.1, 0.7], [0.5, 0.2]
        ])
    elif mode == 1:
        # Hexagonal-ish
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ])
    elif mode == 2:
        # Spread pattern
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.3, 0.0], [0.7, 0.3], [0.0, 0.3],
            [0.33, 0.33],
            [0.1, 0.05], [0.85, 0.05], [0.05, 0.85], [0.5, 0.15]
        ])
    elif mode == 3:
        # Random structured
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [rng.uniform(0.3, 0.7), 0.0],
            [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)],
            [0.0, rng.uniform(0.3, 0.7)],
            [1/3, 1/3],
            [rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)],
            [rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)],
            [rng.uniform(0.1, 0.3), rng.uniform(0.7, 0.9)],
            [rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)]
        ])
    else:
        # Pure random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.01, 0.99)
                v = rng.uniform(0.01, 0.99)
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
    
    # Add small jitter for diversity
    jitter = rng.normal(0, 0.01, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _l_bfgs_optimize(bary0, beta=100.0, maxiter=500, use_soft=True):
    """
    L-BFGS-B optimization.
    """
    try:
        x0 = bary0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        if use_soft:
            obj = lambda x: _softmin_objective(x, beta=beta)
        else:
            obj = _hard_objective
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _beta_anneal_refine(bary0, betas=(30, 60, 120, 250, 500), maxiter_per_stage=200):
    """
    Refine using beta annealing: start with smooth surrogate, increase sharpness.
    """
    bary = bary0.copy()
    for beta in betas:
        bary = _l_bfgs_optimize(bary, beta=beta, maxiter=maxiter_per_stage, use_soft=True)
    return bary

def _nelder_mead_refine(bary0, maxiter=300):
    """
    Nelder-Mead for local refinement on the hard objective.
    """
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def _differential_evolution_global(seed=42, maxiter=50, popsize=15):
    """
    Differential evolution for global search.
    """
    try:
        bounds = [(0, 1)] * (2 * N)
        
        def de_obj(x):
            return _softmin_objective(x, beta=50.0)
        
        de_result = differential_evolution(
            de_obj,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            updating='deferred'
        )
        
        bary_opt = _clamp_bary(de_result.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return None

def _active_set_refine(bary0, maxiter=200):
    """
    Refine by identifying active (binding) triples and using SLSQP with constraints.
    """
    try:
        points = _bary_to_cart(bary0)
        areas = _all_areas(points)
        min_area = np.min(areas)
        
        # Identify active constraints: areas close to min
        tol = max(0.05 * min_area, 1e-6)
        active_mask = areas <= min_area + tol
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return bary0.copy()
        
        # Use SLSQP to equalize active areas
        def objective(x):
            bary = x.reshape(N, 2)
            bary = _clamp_bary(bary)
            points = _bary_to_cart(bary)
            return -_min_area(points)
        
        def constraint_equalize(x, idx):
            """Constraint: area[idx] - min_area <= tol"""
            bary = x.reshape(N, 2)
            bary = _clamp_bary(bary)
            points = _bary_to_cart(bary)
            a = _all_areas(points)
            return a[idx] - np.min(a) - tol
        
        constraints = []
        for idx in active_indices[:10]:  # Limit number of constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=idx: constraint_equalize(x, i)})
        
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            objective, bary0.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _is_time_up():
        return time.time() > deadline - 3.0
    
    best_area = 0.0
    best_bary = None
    
    def _update_best(bary):
        nonlocal best_area, best_bary
        if bary is None:
            return
        area = _score_bary(bary)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
    
    # Phase 1: Differential Evolution for global search
    if not _is_time_up():
        de_result = _differential_evolution_global(seed=42, maxiter=50, popsize=15)
        if de_result is not None:
            _update_best(de_result)
    
    # Phase 2: Multi-start with structured initializations + L-BFGS-B + beta annealing
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    for mode in range(5):
        if _is_time_up():
            break
        
        bary0 = _generate_initial_bary(seed=42 + mode, mode=mode)
        
        # Beta annealing refinement
        if not _is_time_up():
            bary_annealed = _beta_anneal_refine(bary0, betas=(30, 60, 120, 250, 500), maxiter_per_stage=150)
            _update_best(bary_annealed)
        
        # Nelder-Mead refinement on hard objective
        if not _is_time_up():
            bary_nm = _nelder_mead_refine(bary_annealed, maxiter=300)
            _update_best(bary_nm)
    
    # Phase 3: Active set refinement
    if best_bary is not None and not _is_time_up():
        bary_active = _active_set_refine(best_bary, maxiter=200)
        _update_best(bary_active)
    
    # Phase 4: Final polish with high-beta L-BFGS-B
    if best_bary is not None and not _is_time_up():
        bary_polished = _l_bfgs_optimize(best_bary, beta=500, maxiter=300, use_soft=True)
        _update_best(bary_polished)
        
        bary_final = _nelder_mead_refine(bary_polished, maxiter=200)
        _update_best(bary_final)
    
    if best_bary is None:
        # Fallback
        best_bary = _generate_initial_bary(seed=0, mode=0)
    
    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary[i] = [u, v]
                break
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)
