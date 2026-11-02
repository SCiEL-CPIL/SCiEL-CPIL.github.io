"""Heilbronn 11 triangles: Differential Evolution with soft-min smoothing and Nelder-Mead polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_NTRIP = len(_TRIPLETS)

# Barycentric to Cartesian conversion
def _bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    u = np.zeros(N)
    v = np.zeros(N)
    for i in range(N):
        x, y = pts[i]
        v[i] = 2 * y / S3
        u[i] = x - 0.5 * v[i]
    return np.column_stack((u, v))

def _project_to_triangle(bary):
    """Project barycentric coords into the triangle: u>=0, v>=0, u+v<=1."""
    bary = bary.copy()
    u = bary[:, 0]
    v = bary[:, 1]
    u = np.maximum(0, u)
    v = np.maximum(0, v)
    s = u + v
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack((u, v))

def _min_triangle_area_bary(bary):
    """Hard minimum triangle area in barycentric coordinates."""
    u = bary[:, 0]
    v = bary[:, 1]
    u1, v1 = u[_T_I], v[_T_I]
    u2, v2 = u[_T_J], v[_T_J]
    u3, v3 = u[_T_K], v[_T_K]
    dets = (u2*v3 - u3*v2) - u1*(v3 - v2) + v1*(u3 - u2)
    areas = TRI_AREA * np.abs(dets)
    return np.min(areas)

def _soft_min_area(bary, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    Returns a smooth surrogate that is close to the hard min for large beta.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    u1, v1 = u[_T_I], v[_T_I]
    u2, v2 = u[_T_J], v[_T_J]
    u3, v3 = u[_T_K], v[_T_K]
    dets = (u2*v3 - u3*v2) - u1*(v3 - v2) + v1*(u3 - u2)
    areas = TRI_AREA * np.abs(dets)
    # Log-sum-exp for the minimum: min(a) = -max(-a)
    # -max(-a_i) = -(1/beta) * log(sum(exp(-beta*a_i)))
    # Use stable log-sum-exp
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if max_neg == 0:
        return 0.0
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -log_sum / beta
    return soft_min

def _bary_objective(bary, beta=50.0):
    """Objective for optimization: negative soft-min area (to maximize)."""
    bary = _project_to_triangle(bary)
    return -_soft_min_area(bary, beta)

def _penalized_hard_min(bary):
    """Hard minimum area with a penalty for being outside the triangle."""
    bary = _project_to_triangle(bary)
    return -_min_triangle_area_bary(bary)

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area
        bary = _project_to_triangle(bary)
        area = _min_triangle_area_bary(bary)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            return True
        return False

    def _nelder_mead_polish(bary0, max_iter=5000, beta=50.0):
        """Polish a candidate using Nelder-Mead on the soft-min objective."""
        x0 = bary0.flatten()
        
        def obj(x):
            b = x.reshape(-1, 2)
            b = _project_to_triangle(b)
            return -_soft_min_area(b, beta)
        
        try:
            res = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return p
        except Exception:
            return bary0

    def _differential_evolution(bary0, pop_size=30, max_gen=200, F=0.8, CR=0.9):
        """
        Run a simplified Differential Evolution loop in barycentric space.
        """
        pop = np.zeros((pop_size, N, 2))
        # Initialize population around bary0
        pop[0] = bary0
        for i in range(1, pop_size):
            pop[i] = _project_to_triangle(bary0 + rng.normal(0, 0.1, size=(N, 2)))
        
        # Evaluate initial population
        areas = np.array([_min_triangle_area_bary(p) for p in pop])
        best_idx = np.argmax(areas)
        best_bary_local = pop[best_idx].copy()
        best_area_local = areas[best_idx]
        _update_best(best_bary_local)
        
        for gen in range(max_gen):
            if not _elapsed_ok(10):
                break
            
            for i in range(pop_size):
                if not _elapsed_ok(10):
                    break
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = _project_to_triangle(mutant)
                
                # Crossover
                trial = pop[i].copy()
                cross_mask = rng.random((N, 2)) < CR
                # Ensure at least one gene is taken from mutant
                if not np.any(cross_mask):
                    idx = rng.integers(0, N*2)
                    cross_mask.flat[idx] = True
                trial[cross_mask] = mutant[cross_mask]
                trial = _project_to_triangle(trial)
                
                # Selection
                trial_area = _min_triangle_area_bary(trial)
                if trial_area > areas[i]:
                    pop[i] = trial
                    areas[i] = trial_area
                    if trial_area > best_area_local:
                        best_area_local = trial_area
                        best_bary_local = trial.copy()
                        _update_best(best_bary_local)
        
        return best_bary_local, best_area_local

    # Phase 1: Generate diverse initial configurations
    configs = []
    
    # 1. Symmetric base
    base_sym = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
        [1/6, 1/6], [5/6, 1/6], [1/6, 5/6],
        [0.5, 0.0]  # Duplicate, will be perturbed
    ])
    configs.append(base_sym)
    
    # 2. Random uniform points
    for _ in range(10):
        bary = rng.random((N, 2))
        mask = bary[:, 0] + bary[:, 1] > 1
        bary[mask, 0] = 1 - bary[mask, 0]
        bary[mask, 1] = 1 - bary[mask, 1]
        configs.append(_project_to_triangle(bary))
    
    # 3. Perturbed symmetric base
    for _ in range(5):
        pts = base_sym + rng.normal(0, 0.1, size=(N, 2))
        configs.append(_project_to_triangle(pts))

    # Phase 2: Differential Evolution from each config
    for config in configs:
        if not _elapsed_ok(20):
            break
        
        # Run DE for a limited number of generations
        de_best, de_area = _differential_evolution(config, pop_size=20, max_gen=100)
        
        # Polish the best DE result with Nelder-Mead
        if _elapsed_ok(10):
            polished = _nelder_mead_polish(de_best, max_iter=3000)
            _update_best(polished)

    # Phase 3: Final fine polish on the global best
    if best_bary is not None and _elapsed_ok(10):
        # Use a larger beta for a sharper approximation
        polished = _nelder_mead_polish(best_bary, max_iter=5000, beta=100.0)
        _update_best(polished)
        
        # Another round of polish with even larger beta
        if _elapsed_ok(5):
            polished = _nelder_mead_polish(best_bary, max_iter=5000, beta=200.0)
            _update_best(polished)

    if best_bary is None:
        bary = rng.random((N, 2))
        mask = bary[:, 0] + bary[:, 1] > 1
        bary[mask, 0] = 1 - bary[mask, 0]
        bary[mask, 1] = 1 - bary[mask, 1]
        best_bary = _project_to_triangle(bary)

    return _bary_to_cart(best_bary)
