"""Packing 50 equal circles in unit circle: Radius Bisection + Penalty Optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
# Safety margin: Evaluator kills at 150s. We aim to finish by 135s.
DEADLINE = 135.0

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the maximum radius r for which circles are disjoint and inside unit circle."""
    n = centers.shape[0]
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _penalty_obj_and_grad(params, n, r, penalty_factor=1000.0):
    """
    Objective: Sum of squared violations for constraints:
    1. Pairwise distance >= 2r
    2. Boundary distance >= r (i.e., |c_i| <= 1-r)
    
    Returns (objective, gradient).
    """
    centers = params.reshape(n, 2)
    r_max = 1.0 - r  # Max norm allowed
    
    # 1. Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Violation: 2r - d_ij if positive
    viol_pairs = np.maximum(2.0 * r - pair_dists, 0.0)
    
    # 2. Boundary constraints
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    # Violation: norm - r_max if positive
    viol_bound = np.maximum(norms - r_max, 0.0)
    
    # Objective: Sum of squared violations * penalty_factor
    obj_val = penalty_factor * (np.sum(viol_pairs ** 2) + np.sum(viol_bound ** 2))
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pair gradients
    # d/dc_i ( (2r - d)^2 ) = 2(2r-d) * (-1) * (c_i - c_j)/d
    # We only care where violation > 0
    mask_pairs = viol_pairs > 0
    if np.any(mask_pairs):
        idx = np.where(mask_pairs)[0]
        ii = i_idx[idx]
        jj = j_idx[idx]
        dd = pair_dists[idx]
        vv = viol_pairs[idx]
        
        valid = dd > 1e-8
        if np.any(valid):
            ii_v = ii[valid]
            jj_v = jj[valid]
            dd_v = dd[valid]
            vv_v = vv[valid]
            
            # Gradient term: 2 * penalty * vv * (-1) * vec / d
            # vec = c_i - c_j
            vecs = (centers[ii_v] - centers[jj_v]) / dd_v[:, np.newaxis]
            coeff = 2.0 * penalty_factor * vv_v[:, np.newaxis]
            
            contrib = coeff * vecs
            np.add.at(grad, ii_v, -contrib)
            np.add.at(grad, jj_v, contrib)
            
    # Boundary gradients
    # d/dc ( (|c| - r_max)^2 ) = 2(|c|-r_max) * (c/|c|)
    mask_bound = viol_bound > 0
    if np.any(mask_bound):
        idx_b = np.where(mask_bound)[0]
        valid_b = norms[idx_b] > 1e-8
        if np.any(valid_b):
            idx_v = idx_b[valid_b]
            vv_v = viol_bound[idx_v]
            norms_v = norms[idx_v]
            centers_v = centers[idx_v]
            
            vecs = centers_v / norms_v[:, np.newaxis]
            coeff = 2.0 * penalty_factor * vv_v[:, np.newaxis]
            
            contrib = coeff * vecs
            grad[idx_v] += contrib

    return obj_val, grad

def _try_feasibility(centers_init, r, maxiter=50, penalty_factor=1000.0):
    """
    Try to find a valid configuration for radius r starting from centers_init.
    Returns (success, final_centers, final_score).
    """
    try:
        res = minimize(
            _penalty_obj_and_grad,
            centers_init.flatten(),
            args=(N, r, penalty_factor),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if np.all(np.isfinite(res.x)):
            final_centers = res.x.reshape(N, 2)
            # Check if actual score is at least r (with small tolerance)
            score = _compute_score(final_centers)
            # If score is close to r, consider it feasible
            if score >= r - 1e-5:
                return True, final_centers, score
            else:
                return False, final_centers, score
    except Exception:
        pass
    return False, centers_init, _compute_score(centers_init)

def _get_ring_init(n):
    """
    Initialize based on the known optimal structure: 1+6+12+18+13.
    """
    centers = [[0.0, 0.0]]
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        a = 2 * np.pi * i / 6
        centers.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 12 circles
    r2 = 0.45
    for i in range(12):
        a = 2 * np.pi * i / 12
        centers.append([r2 * np.cos(a), r2 * np.sin(a)])
    # Ring 3: 18 circles
    r3 = 0.65
    for i in range(18):
        a = 2 * np.pi * i / 18
        centers.append([r3 * np.cos(a), r3 * np.sin(a)])
    # Ring 4: 13 circles (50-1-6-12-18 = 13)
    r4 = 0.85
    for i in range(13):
        a = 2 * np.pi * i / 13
        centers.append([r4 * np.cos(a), r4 * np.sin(a)])
        
    return np.array(centers[:n])

def _get_random_init(n, seed):
    """Random initialization inside unit circle."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers)

def construct():
    start_time = time.time()
    
    # Initial guesses for radius
    # Best known is ~0.1258. We can start lower to be safe, e.g., 0.10
    r_low = 0.05
    r_high = 0.13  # Slightly above best known to allow improvement
    
    best_centers = _get_ring_init(N)
    best_score = _compute_score(best_centers)
    
    # Pre-compute initializations to reuse in bisection
    inits = [
        _get_ring_init(N),
        _get_random_init(N, 1),
        _get_random_init(N, 42),
        _get_random_init(N, 123),
    ]
    
    # Bisection loop
    # We want to find the max r such that feasibility holds.
    # If r is feasible, r_low = r. Else r_high = r.
    
    # To save time, we limit the number of bisection steps.
    # Each step involves trying multiple inits.
    max_steps = 20
    
    for step in range(max_steps):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        feasible = False
        best_feasible_centers = None
        best_feasible_score = 0.0
        
        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Ensure init is inside container for radius r_mid
            # Scale init to fit within 1 - r_mid
            max_norm = 1.0 - r_mid
            norms = np.linalg.norm(init, axis=1)
            if np.any(norms > max_norm):
                init = init.copy()
                mask = norms > max_norm
                init[mask] *= max_norm / norms[mask]
                
            success, centers, score = _try_feasibility(init, r_mid, maxiter=30)
            
            if success:
                feasible = True
                if score > best_feasible_score:
                    best_feasible_score = score
                    best_feasible_centers = centers.copy()
                
        if feasible:
            r_low = r_mid
            if best_feasible_centers is not None and best_feasible_score > best_score:
                best_score = best_feasible_score
                best_centers = best_feasible_centers.copy()
        else:
            r_high = r_mid
            
        # Convergence check
        if r_high - r_low < 1e-5:
            break

    # Final polish: Try to improve the best found centers slightly
    # We can try a local optimization on the "max min" objective for the best centers found
    # But since we used penalty method, the centers should be good.
    # Let's just return the best centers found.
    
    # Ensure strict validity
    norms = np.linalg.norm(best_centers, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_centers = best_centers.copy()
        best_centers[mask] *= 0.9999 / norms[mask]
        
    return best_centers

# EVOLVE-BLOCK-END
