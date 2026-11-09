"""
Strategy: Smooth Surrogate Optimization with Nelder-Mead Simplex
Instead of stochastic Simulated Annealing, we use a deterministic local search method.
We model the problem as maximizing a smooth surrogate: the soft-min (log-sum-exp) of
clearances. This smooths out the non-differentiable kinks of the true min-clearance
objective, allowing Nelder-Mead to navigate the landscape more effectively.
We run many quick Nelder-Mead optimizations from diverse initial configurations,
then apply longer, more refined Nelder-Mead polishes to the top candidates.
"""

import numpy as np
import time
import random

N = 50
DEADLINE = 145.0  # Safety margin from 150s

def _radius_score(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^2."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distance
    dist_to_boundary = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2
    
    return min(min_boundary, min_pairwise)

def _compute_clearances(centers):
    """
    Compute all clearances: boundary distances and half pairwise distances.
    Returns a tuple (min_clearance, all_clearances_array).
    """
    # Boundary clearances
    b_x_pos = centers[:, 0] + 0.5
    b_x_neg = 0.5 - centers[:, 0]
    b_y_pos = centers[:, 1] + 0.5
    b_y_neg = 0.5 - centers[:, 1]
    
    boundary_clearances = np.concatenate([b_x_pos, b_x_neg, b_y_pos, b_y_neg])
    
    # Pairwise clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    pair_clearances = dists[iu] / 2.0
    
    all_clearances = np.concatenate([boundary_clearances, pair_clearances])
    min_clearance = np.min(all_clearances)
    
    return min_clearance, all_clearances

def _soft_min_objective(x, beta=100.0):
    """
    Smooth surrogate objective: maximize soft-min of clearances.
    x is a flattened array of shape (N*2,).
    Uses log-sum-exp for numerical stability.
    """
    centers = x.reshape(N, 2)
    
    # Boundary clearances
    b_x_pos = centers[:, 0] + 0.5
    b_x_neg = 0.5 - centers[:, 0]
    b_y_pos = centers[:, 1] + 0.5
    b_y_neg = 0.5 - centers[:, 1]
    
    boundary_clearances = np.concatenate([b_x_pos, b_x_neg, b_y_pos, b_y_neg])
    
    # Pairwise clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    pair_clearances = dists[iu] / 2.0
    
    all_clearances = np.concatenate([boundary_clearances, pair_clearances])
    
    # Soft min: -softmax(-clearances) = -logsumexp(-beta * clearances) / beta
    # For numerical stability, use logsumexp trick
    neg_beta_c = -beta * all_clearances
    max_val = np.max(neg_beta_c)
    if max_val > 500:  # Avoid overflow
        return 0.0
    log_sum = np.log(np.sum(np.exp(neg_beta_c - max_val))) + max_val
    soft_min_val = -log_sum / beta
    
    return soft_min_val

def _nelder_mead_optimize(initial_centers, time_budget=5.0, beta=100.0, max_iter=2000):
    """
    Run Nelder-Mead optimization on the soft-min surrogate.
    Returns (best_centers, best_radius).
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    x0 = initial_centers.flatten()
    
    best_x = x0.copy()
    best_f = _soft_min_objective(x0, beta)
    best_radius = _radius_score(initial_centers)
    
    def _objective(x):
        nonlocal best_x, best_f, best_radius
        f = _soft_min_objective(x, beta)
        if f > best_f:
            best_f = f
            best_x = x.copy()
            # Periodically check true radius
            if time.time() - start_time < time_budget - 0.1:
                r = _radius_score(x.reshape(N, 2))
                if r > best_radius:
                    best_radius = r
        return -f  # Minimize negative soft-min
    
    # Nelder-Mead options
    options = {
        'maxiter': max_iter,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', options=options)
        
        # Final evaluation
        final_radius = _radius_score(result.x.reshape(N, 2))
        if final_radius > best_radius:
            best_radius = final_radius
            best_x = result.x.copy()
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        # Return best found so far
        pass
    
    best_centers = best_x.reshape(N, 2)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers, best_radius

def _construct():
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Grid (7x8)
    m, n = 7, 8
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 2. Hexagonal
    points = []
    row_height = 0.09
    col_width = 0.1
    offset = col_width / 2
    for row in range(8):
        y = -0.5 + 0.05 + row * row_height
        for col in range(7):
            if len(points) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 3. Random inits (15)
    for _ in range(15):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
    
    # Phase 1: Quick Nelder-Mead polishes on all inits
    # Allocate time: ~2s per init, 17 inits = 34s
    time_per_init_phase1 = 2.0
    max_iter_phase1 = 500
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 10.0:
            break
            
        remaining = DEADLINE - time.time()
        if remaining < 3.0:
            break
            
        t_budget = min(time_per_init_phase1, remaining - 2.0)
        
        optimized, r = _nelder_mead_optimize(init, time_budget=t_budget, beta=50.0, max_iter=max_iter_phase1)
        
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
    
    # Phase 2: Longer Nelder-Mead on top candidates
    # Keep track of top 5 candidates from Phase 1
    # Re-run Phase 1 to collect candidates, or just use best_centers
    # For simplicity, run longer NM on best_centers and a few perturbed versions
    
    if best_centers is not None and time.time() < DEADLINE - 15.0:
        # Collect candidates: best_centers + 4 perturbed versions
        candidates = [best_centers.copy()]
        
        for _ in range(4):
            perturbed = best_centers + rng.normal(0, 0.01, size=(N, 2))
            perturbed = np.clip(perturbed, -0.5, 0.5)
            candidates.append(perturbed)
        
        time_per_candidate = 5.0
        max_iter_phase2 = 2000
        
        for cand in candidates:
            if time.time() > DEADLINE - 5.0:
                break
                
            remaining = DEADLINE - time.time()
            if remaining < 3.0:
                break
                
            t_budget = min(time_per_candidate, remaining - 2.0)
            
            optimized, r = _nelder_mead_optimize(cand, time_budget=t_budget, beta=100.0, max_iter=max_iter_phase2)
            
            if r > best_radius:
                best_radius = r
                best_centers = optimized.copy()
    
    # Phase 3: Final high-precision polish
    if best_centers is not None and time.time() < DEADLINE - 3.0:
        remaining = DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            polished, r = _nelder_mead_optimize(best_centers, time_budget=min(5.0, remaining), 
                                                beta=200.0, max_iter=3000)
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()
    
    if best_centers is None:
        # Fallback: Grid
        m, n = 7, 8
        points = []
        for i in range(n):
            for j in range(m):
                if len(points) >= N:
                    break
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / n - 0.5
                points.append([x, y])
        best_centers = np.array(points[:N])
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

def construct():
    return _construct()
