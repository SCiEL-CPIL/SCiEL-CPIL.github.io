"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the radius for a set of centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _neg_score(centers):
    return -_score(centers)

def _jacobian(centers):
    """Compute gradient of -score w.r.t. centers."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Find the minimum pairwise distance
    min_pair_val = np.min(dists)
    min_pair_idx = np.argmin(dists)
    i, j = np.unravel_index(min_pair_idx, (n, n))
    
    # Gradient of -d_{ij}/2 w.r.t. c_i and c_j
    if dists[i, j] > 1e-15:
        unit_vec = diff[i, j] / dists[i, j]
        grad[i] += unit_vec / 2.0
        grad[j] -= unit_vec / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_val = np.min(wall_dists)
    min_wall_idx = np.argmin(wall_dists)
    i, k = np.unravel_index(min_wall_idx, (n, 2))
    
    # Gradient of -(0.5 - |c_{ik}|) = sign(c_{ik})
    # We want to maximize the minimum, so we push away from the wall
    # d/dc_{ik} (0.5 - |c_{ik}|) = -sign(c_{ik})
    # So d/dc_{ik} (-(0.5 - |c_{ik}|)) = sign(c_{ik})
    if centers[i, k] > 1e-15:
        grad[i, k] += 1.0
    elif centers[i, k] < -1e-15:
        grad[i, k] -= 1.0
    else:
        grad[i, k] += 1.0  # Arbitrary push
    
    # The actual gradient of -score is the gradient of the active constraint
    # But we need to be careful: if min_pair < min_wall, the active constraint is pair
    # Otherwise it's wall
    
    # Recompute to determine which is active
    min_pair = np.min(dists) / 2.0
    min_wall = np.min(wall_dists)
    
    grad = np.zeros_like(centers)
    if min_pair <= min_wall:
        # Active constraint is pair (i, j)
        if dists[i, j] > 1e-15:
            unit_vec = diff[i, j] / dists[i, j]
            grad[i] += unit_vec / 2.0
            grad[j] -= unit_vec / 2.0
    else:
        # Active constraint is wall at (i, k)
        if centers[i, k] > 1e-15:
            grad[i, k] += 1.0
        elif centers[i, k] < -1e-15:
            grad[i, k] -= 1.0
        else:
            grad[i, k] += 1.0
    
    return grad

def _soft_min_score(centers, beta=100.0):
    """Smooth surrogate using log-sum-exp for the minimum."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all constraints
    pair_vals = dists / 2.0
    wall_vals = wall_dists
    
    all_vals = np.concatenate([pair_vals.flatten(), wall_vals.flatten()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    # Soft min: -logsumexp(-beta * vals) / beta
    max_val = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    soft_min = max_val - np.log(log_sum) / beta
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Try multiple random initializations and local optimization
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            cur = centers.copy()
        else:
            # Random perturbation of grid
            cur = centers.copy()
            cur += np.random.normal(0, 0.02, size=cur.shape)
            # Clip to valid range
            cur = np.clip(cur, -0.48, 0.48)
        
        # Local optimization using SLSQP
        for step in range(3):
            if time.time() > deadline:
                break
            
            result = minimize(
                _neg_score,
                cur,
                jac=_jacobian,
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < _neg_score(cur):
                cur = result.x
                cur = np.clip(cur, -0.49, 0.49)
        
        cur_score = _score(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
        
        # If we've found a good score, do a final polish
        if best_score > 0.071:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        result = minimize(
            _neg_score,
            best_centers,
            jac=_jacobian,
            method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if result.fun < _neg_score(best_centers):
            best_centers = np.clip(result.x, -0.49, 0.49)
    
    return best_centers
# EVOLVE-BLOCK-END
