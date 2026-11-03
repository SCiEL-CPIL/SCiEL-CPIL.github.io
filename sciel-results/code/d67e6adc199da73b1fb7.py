"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (C(11,3) = 165)
# Storing as a single array for vectorized operations
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I_IDX = _TRIPLES[:, 0]
_J_IDX = _TRIPLES[:, 1]
_K_IDX = _TRIPLES[:, 2]

def _bary_to_cart(bary):
    """Convert barycentric coordinates (lambda1, lambda2) to Cartesian coordinates.
    lambda3 is implicit: 1 - lambda1 - lambda2.
    Point = lambda1 * V1 + lambda2 * V2 + lambda3 * V0
    Since V0 is (0,0), Point = lambda1 * V1 + lambda2 * V2
    """
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Ensure barycentric coordinates are valid (non-negative, sum <= 1)."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_triangle_areas(points):
    """Compute all triangle areas formed by triples of points."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Compute the minimum triangle area."""
    return np.min(_all_triangle_areas(points))

def _softmin_objective(x, beta=100.0):
    """
    Objective function using soft-min (log-sum-exp) approximation of the minimum area.
    x is a flat array of barycentric coordinates (size 2*N).
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _all_triangle_areas(points)
    
    # Soft min: min(a) ~ min(a) - (1/beta) * log(sum(exp(-beta * (a - min(a)))))
    # This is smooth and differentiable.
    min_area = np.min(areas)
    # Numerical stability: shift by min_area
    shifted = areas - min_area
    # Avoid overflow in exp
    # exp(-beta * shifted) is safe if shifted >= 0
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area - log_sum / beta
    return -soft_min

def _hard_objective(x):
    """Objective function using the true minimum area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area(points)

def _generate_structural_initialization(mode, rng):
    """Generate diverse initial configurations based on symmetry and heuristics."""
    if mode == 0:
        # Symmetric: Vertices, Midpoints, Centroid, + 4 others
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.5, 0.1]
        ])
    elif mode == 1:
        # Hexagonal-ish packing
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ])
    elif mode == 2:
        # Random spread
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v < 1.0:
                    bary[i] = [u, v]
                    break
    else:
        # Pure random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v < 1.0:
                    bary[i] = [u, v]
                    break
    
    # Add small jitter to break symmetry if needed
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _refine_with_annealing(bary0, time_limit=10.0):
    """
    Refine a configuration using beta annealing with L-BFGS-B.
    Starts with smooth objective (low beta) and increases sharpness.
    """
    start_time = time.time()
    bary = bary0.copy()
    
    # Stages: (beta, maxiter)
    # Low beta = smooth, easier to navigate global landscape
    # High beta = sharp, approaches true min
    stages = [
        (20, 50),
        (50, 100),
        (100, 200),
        (200, 300)
    ]
    
    bounds = [(0, 1)] * (2 * N)
    
    for beta, maxiter in stages:
        if time.time() - start_time > time_limit:
            break
            
        try:
            x0 = bary.flatten()
            res = minimize(
                _softmin_objective, x0, args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            bary = _clamp_bary(res.x.reshape(N, 2))
        except Exception:
            pass
            
    return bary

def _final_polish(bary0, time_limit=5.0):
    """
    Final polish using Nelder-Mead on the hard objective.
    """
    start_time = time.time()
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (or 150s if strict)
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Global Search with Differential Evolution
    # DE is good for exploring the space. We run it for a short time to get good candidates.
    # We use the soft-min objective with a moderate beta for DE.
    
    if time_left() > 40.0:
        try:
            bounds = [(0, 1)] * (2 * N)
            def de_obj(x):
                return _softmin_objective(x, beta=50.0)
            
            de_result = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20, # Limited iterations
                popsize=10,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                updating='deferred'
            )
            bary_de = _clamp_bary(de_result.x.reshape(N, 2))
            
            # Refine DE result
            bary_refined = _refine_with_annealing(bary_de, time_limit=10.0)
            score = -_hard_objective(bary_refined.flatten())
            if score > best_score:
                best_score = score
                best_bary = bary_refined
        except Exception:
            pass

    # 2. Local Optimization from Structured Initializations
    # These provide diverse, high-quality starting points that might be missed by DE.
    
    initializations = []
    for mode in range(4):
        bary_init = _generate_structural_initialization(mode, rng)
        initializations.append(bary_init)
        
    # Add a few random initializations
    for _ in range(3):
        bary_init = _generate_structural_initialization(3, rng)
        initializations.append(bary_init)

    for bary_init in initializations:
        if time_left() < 15.0:
            break
            
        # Refine with annealing
        bary_refined = _refine_with_annealing(bary_init, time_limit=8.0)
        
        # Final polish
        if time_left() > 5.0:
            bary_polished = _final_polish(bary_refined, time_limit=4.0)
        else:
            bary_polished = bary_refined
            
        score = -_hard_objective(bary_polished.flatten())
        if score > best_score:
            best_score = score
            best_bary = bary_polished

    # 3. If we still have time, do a few more random restarts with local optimization
    # This helps escape local minima.
    while time_left() > 10.0:
        bary_init = _generate_structural_initialization(3, rng)
        bary_refined = _refine_with_annealing(bary_init, time_limit=5.0)
        if time_left() > 3.0:
            bary_polished = _final_polish(bary_refined, time_limit=2.0)
        else:
            bary_polished = bary_refined
            
        score = -_hard_objective(bary_polished.flatten())
        if score > best_score:
            best_score = score
            best_bary = bary_polished

    if best_bary is None:
        # Fallback: Centroid and vertices
        best_bary = np.zeros((N, 2))
        best_bary[0] = [0.0, 0.0]
        best_bary[1] = [1.0, 0.0]
        best_bary[2] = [0.0, 1.0]
        best_bary[3] = [0.5, 0.5]
        for i in range(4, N):
            best_bary[i] = [1/3, 1/3] + rng.normal(0, 0.1, 2)
        best_bary = _clamp_bary(best_bary)

    points = _bary_to_cart(best_bary)
    return np.array(points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
