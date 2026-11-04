"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum radius for the given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    # Distance to walls
    # w_i = min(0.5 - |c_ik|)
    dists_wall = 0.5 - np.abs(centers)
    min_wall = np.min(dists_wall, axis=1)
    
    # Pairwise distances
    # Use broadcasting for simplicity, N=100 is small
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    
    r = np.min(np.min(min_pair), np.min(min_wall))
    return r

def _soft_min_obj_grad(x_flat, beta):
    """
    Objective: Minimize -SoftMin(beta, distances)
    Equivalent to Maximizing SoftMin.
    x_flat: flattened centers (N*3,)
    """
    centers = x_flat.reshape(N, 3)
    
    # Compute pairwise distances and gradients
    # d_ij = |c_i - c_j|
    # We need sum over j != i of exp(-beta * d_ij) * (c_i - c_j) / d_ij
    
    # Vectorized computation
    # diffs[i, j] = c_i - c_j
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Avoid division by zero
    # Set diagonal to 1 to avoid 0/0, then mask out
    np.fill_diagonal(dists, 1.0)
    
    # exp_term[i, j] = exp(-beta * dists[i, j])
    exp_terms = np.exp(-beta * dists)
    np.fill_diagonal(exp_terms, 0.0)
    
    # grad_dist[i, j] = (c_i - c_j) / dists[i, j]
    # Note: dists is (N, N), diffs is (N, N, 3)
    # We need to divide diffs by dists[..., np.newaxis]
    grad_dists = diffs / dists[..., np.newaxis]
    
    # Contribution to gradient from pairs:
    # grad_pair[i] = sum_j exp_term[i, j] * grad_dist[i, j]
    # Note: The formula derived was:
    # grad J / grad x_m = 1/S * ( sum_j exp(-beta d_mj) * (x_m - x_j)/d_mj + ... )
    # So we sum over j for each i.
    grad_pairs = np.sum(exp_terms[..., np.newaxis] * grad_dists, axis=1)
    
    # Wall contributions
    # Walls:
    # 0: x = 0.5, w = 0.5 - x, n = (-1, 0, 0)
    # 1: x = -0.5, w = 0.5 + x, n = (1, 0, 0)
    # 2: y = 0.5, w = 0.5 - y, n = (0, -1, 0)
    # 3: y = -0.5, w = 0.5 + y, n = (0, 1, 0)
    # 4: z = 0.5, w = 0.5 - z, n = (0, 0, -1)
    # 5: z = -0.5, w = 0.5 + z, n = (0, 0, 1)
    
    w_x_pos = 0.5 - centers[:, 0]
    w_x_neg = 0.5 + centers[:, 0]
    w_y_pos = 0.5 - centers[:, 1]
    w_y_neg = 0.5 + centers[:, 1]
    w_z_pos = 0.5 - centers[:, 2]
    w_z_neg = 0.5 + centers[:, 2]
    
    exp_w_x_pos = np.exp(-beta * w_x_pos)
    exp_w_x_neg = np.exp(-beta * w_x_neg)
    exp_w_y_pos = np.exp(-beta * w_y_pos)
    exp_w_y_neg = np.exp(-beta * w_y_neg)
    exp_w_z_pos = np.exp(-beta * w_z_pos)
    exp_w_z_neg = np.exp(-beta * w_z_neg)
    
    # Sum of wall exponentials
    S_walls = exp_w_x_pos + exp_w_x_neg + exp_w_y_pos + exp_w_y_neg + exp_w_z_pos + exp_w_z_neg
    
    # Gradient contribution from walls
    # grad_wall[i] = exp_w_x_pos[i] * (-1, 0, 0) + exp_w_x_neg[i] * (1, 0, 0) + ...
    grad_walls = np.zeros((N, 3))
    grad_walls[:, 0] = -exp_w_x_pos + exp_w_x_neg
    grad_walls[:, 1] = -exp_w_y_pos + exp_w_y_neg
    grad_walls[:, 2] = -exp_w_z_pos + exp_w_z_neg
    
    # Total S
    S_pairs = np.sum(exp_terms)
    S = S_pairs + np.sum(S_walls)
    
    # If S is very small (large beta, large distances), underflow might occur.
    # But we are maximizing distance, so distances should be reasonable.
    if S < 1e-300:
        # Fallback to zero gradient or numerical?
        # Return 0 gradient to avoid NaNs
        return 0.0, np.zeros_like(x_flat)
    
    # Gradient of J = -1/beta ln S
    # dJ/dx = -1/beta * (1/S) * dS/dx
    # dS/dx = -beta * (sum exp * grad_dist + sum exp * grad_w)
    # So dJ/dx = (1/S) * (sum exp * grad_dist + sum exp * grad_w)
    
    total_grad = (grad_pairs + grad_walls) / S
    
    # Objective value J
    # To avoid log(0) if S is tiny, add epsilon?
    # S is sum of exps, so S > 0.
    obj_val = -1.0 / beta * np.log(S)
    
    # Flatten gradient
    grad_flat = total_grad.flatten()
    
    return obj_val, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial configuration: Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N].copy()
    
    # Add small jitter to break symmetry
    rng = np.random.default_rng(42)
    init_centers += rng.uniform(-0.01, 0.01, size=init_centers.shape)
    
    # Clip to bounds
    init_centers = np.clip(init_centers, -0.49, 0.49)
    
    best_centers = init_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Beta annealing schedule
    betas = [10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
    
    current_centers = init_centers.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run optimization
        # We want to maximize SoftMin, so we minimize -SoftMin.
        # _soft_min_obj_grad returns (obj, grad) for the minimization problem.
        
        # Set options
        options = {
            'maxiter': 500,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'disp': False
        }
        
        try:
            res = minimize(
                _soft_min_obj_grad, 
                current_centers.flatten(), 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options=options
            )
            
            if res.success or res.fun < 1e10: # Check for overflow issues
                new_centers = res.x.reshape(N, 3)
                # Clip just in case
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                # Evaluate true radius
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                
                # Update current centers for next beta
                # Use the result from the optimizer
                current_centers = new_centers.copy()
                
        except Exception as e:
            # If optimization fails, keep previous best
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
        # Optional: Restart from best if stuck?
        # For now, just continue with annealing.
        
    # Final check and return
    # Ensure valid shape and values
    if not np.all(np.isfinite(best_centers)):
        # Fallback to simple grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
