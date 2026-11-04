"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _compute_score(centers):
    """
    Compute the score for a given set of centers.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns -1 if invalid.
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
    
    # Check strict inside
    if np.any(centers <= -0.5) or np.any(centers >= 0.5):
        return -1.0
    
    sq_dists = np.sum(centers**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * centers @ centers.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    min_pair_dist = np.min(dists)
    max_abs_coord = np.max(np.abs(centers))
    min_wall_dist = 0.5 - max_abs_coord
    
    score = min(min_pair_dist / 2.0, min_wall_dist)
    return score

def _smooth_min_distance(centers, beta=10.0):
    """
    Smooth approximation of the minimum pairwise distance using log-sum-exp.
    We want to maximize min(d_ij). This is equivalent to minimizing -min(d_ij).
    A smooth lower bound for min(x) is -(1/beta) * log(sum(exp(-beta * x))).
    So we minimize (1/beta) * log(sum(exp(-beta * d_ij))).
    """
    sq_dists = np.sum(centers**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * centers @ centers.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    val = -beta * dists
    max_val = np.max(val)
    log_sum_exp = max_val + np.log(np.sum(np.exp(val - max_val)))
    
    return log_sum_exp / beta

def _neg_smooth_score(centers_flat, beta=10.0):
    centers = centers_flat.reshape(-1, 3)
    return _smooth_min_distance(centers, beta)

def construct():
    # 1. Initial Guess: Jittered Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle to break symmetry
    np.random.shuffle(grid)
    initial_centers = grid[:N]
    
    # Add small jitter
    initial_centers += np.random.normal(0, 0.01, size=(N, 3))
    initial_centers = np.clip(initial_centers, -0.49, 0.49)
    
    # 2. Local Refinement: Multi-start L-BFGS-B with smooth objective
    eps = 1e-5
    bounds = [(-0.5 + eps, 0.5 - eps)] * (N * 3)
    
    best_centers = initial_centers
    best_score = _compute_score(best_centers)
    
    starts = 10
    
    for i in range(starts):
        if i == 0:
            start = initial_centers
        else:
            start = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        start_flat = start.flatten()
        
        try:
            res = minimize(
                _neg_smooth_score,
                start_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20}
            )
            center_candidate = res.x.reshape(-1, 3)
            score_candidate = _compute_score(center_candidate)
            
            if score_candidate > best_score:
                best_score = score_candidate
                best_centers = center_candidate
        except Exception:
            continue
    
    # 3. Final Polish: Subgradient ascent on the exact min distance
    current_centers = best_centers.copy()
    current_score = best_score
    
    for step in range(500):
        sq_dists = np.sum(current_centers**2, axis=1)
        dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * current_centers @ current_centers.T
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(np.maximum(dists_sq, 0))
        
        min_pair_dist = np.min(dists)
        max_abs_coord = np.max(np.abs(current_centers))
        min_wall_dist = 0.5 - max_abs_coord
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > current_score:
            current_score = current_r
            best_centers = current_centers.copy()
        
        if min_pair_dist / 2.0 < min_wall_dist:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            d_ij = dists[i, j]
            if d_ij > 1e-9:
                diff = current_centers[i] - current_centers[j]
                grad_i = diff / d_ij
                grad_j = -diff / d_ij
                
                step_size = 0.01
                current_centers[i] += step_size * grad_i
                current_centers[j] += step_size * grad_j
        else:
            flat_idx = np.argmax(np.abs(current_centers))
            i, k = np.unravel_index(flat_idx, current_centers.shape)
            c_val = current_centers[i, k]
            
            step_size = 0.01
            current_centers[i, k] -= np.sign(c_val) * step_size
        
        current_centers = np.clip(current_centers, -0.4999, 0.4999)

    final_score = _compute_score(best_centers)
    if final_score < 0:
        return np.clip(initial_centers, -0.49, 0.49)
    
    return best_centers
# EVOLVE-BLOCK-END
