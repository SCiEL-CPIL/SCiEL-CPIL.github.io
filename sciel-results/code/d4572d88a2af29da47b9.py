"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy(x):
    """
    Repulsive energy function.
    x is a flat array of size N*3.
    We minimize the sum of inverse squared distances between pairs and to walls.
    This encourages equal spacing (maximizing minimum distance).
    """
    centers = x.reshape(N, 3)
    
    # Pairwise repulsion
    # diff[i, j, k] is the difference in coordinate k between center i and j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_energy = np.sum(1.0 / dists_sq[iu])
    
    # Wall repulsion
    # Distance to each wall for each center
    # Wall distances are 0.5 - x, 0.5 + x, etc.
    # For each center c, wall_dist is min(0.5-c, 0.5+c) for each axis.
    # Actually, distance to nearest wall in axis k is 0.5 - |c_k|.
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-12)
    wall_energy = np.sum(1.0 / wall_dists**2)
    
    # Weight for wall energy to ensure points stay away from walls
    # If wall energy is too low, points might hug walls.
    # If too high, they might cluster in center.
    # 1.0 is a reasonable start.
    return pair_energy + 1.0 * wall_energy

def _energy_grad(x):
    """
    Gradient of the energy function.
    """
    centers = x.reshape(N, 3)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Gradient of 1/d^2 is -2 * diff / d^4
    # Sum over j for each i
    # Force on i from j: -2 * (c_i - c_j) / |c_i - c_j|^4
    # We can compute this efficiently.
    inv_d4 = 1.0 / (dists_sq**2)
    # Zero out diagonal to avoid self-interaction
    np.fill_diagonal(inv_d4, 0.0)
    
    # grad_i = -2 * sum_j (c_i - c_j) * inv_d4[i, j]
    # Note: diff[i, j] = c_i - c_j
    # So force on i is -2 * sum_j diff[i, j] * inv_d4[i, j]
    # Wait, derivative of (c_i - c_j)^2 is 2(c_i - c_j).
    # d/dc_i (1/|c_i-c_j|^2) = -1/|c_i-c_j|^4 * 2(c_i - c_j)
    # So force is -2 * diff * inv_d4.
    # Let's verify signs.
    # If c_i > c_j, diff > 0. Force should push i away from j (positive direction).
    # -2 * positive * positive = negative. This pushes i towards j?
    # Energy is 1/d^2. Gradient points in direction of increasing energy.
    # We want to minimize energy, so we move in direction of -Gradient.
    # If c_i > c_j, distance decreases if i moves left (towards j). Energy increases.
    # So Gradient should be negative (pointing left).
    # My formula: -2 * (c_i - c_j) * inv_d4. If c_i > c_j, result is negative. Correct.
    
    grad -= 2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    # Energy term: 1/(0.5 - |c_k|)^2
    # Let w = 0.5 - |c|. dE/dc = -1/w^2 * dw/dc
    # dw/dc = -sign(c)
    # dE/dc = -1/w^2 * (-sign(c)) = sign(c)/w^2
    # We want to minimize, so move in direction -dE/dc = -sign(c)/w^2
    # This pushes c towards 0 (center) if c>0, and towards 0 if c<0.
    # Wait, if c > 0, wall is at 0.5. w = 0.5 - c.
    # If c increases, w decreases, Energy increases. Gradient is positive.
    # Force (-Grad) is negative, pushes c down (towards 0). Correct.
    
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-12)
    signs = np.sign(centers)
    # Handle c=0 carefully, sign is 0, gradient 0.
    grad -= signs / (wall_dists**2)
    
    return grad.flatten()

def _optimize(centers, time_limit=5.0):
    """
    Run L-BFGS-B optimization on the energy function.
    """
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # L-BFGS-B is good for bounded problems
    # We set a time limit by checking in the callback
    state = {'stop': False}
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            state['stop'] = True
            return True
        return False
    
    try:
        res = minimize(_energy, x0, method='L-BFGS-B', jac=_energy_grad, bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback})
        if state['stop']:
             # If stopped by time, use the last xk? 
             # minimize returns the last iteration if interrupted? 
             # Actually, if callback returns True, it stops. res.x is the last point.
             pass
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = centers
    
    # Ensure bounds are respected strictly
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Initializations
    # 1. Grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_full = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # 2. Fibonacci Sphere projected to cube? Or just random.
    # Let's use multiple restarts with different initializations.
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        
        # Choose initialization strategy
        if restarts % 3 == 1:
            # Grid + Jitter
            idx = rng.permutation(len(grid_full))[:N]
            centers = grid_full[idx].copy()
            centers += rng.normal(0, 0.02, centers.shape)
        elif restarts % 3 == 2:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        else:
            # Clustered Random (start with some structure)
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            # Sort by x, y, z to create a loose grid structure?
            # Just shuffle for now.
            
        centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize
        time_left = deadline - time.time()
        opt_time = min(5.0, time_left / 2.0) # Leave time for scoring and next restart
        
        if opt_time > 0.5:
            refined_centers = _optimize(centers, time_limit=opt_time)
        else:
            refined_centers = centers
            
        score = _score(refined_centers)
        
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()
            
    # Final Polish on the best candidate with remaining time
    if best_centers is not None:
        time_left = deadline - time.time()
        if time_left > 1.0:
            final_time = time_left - 0.5
            refined_centers = _optimize(best_centers, time_limit=final_time)
            score = _score(refined_centers)
            if score > best_score:
                best_score = score
                best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
