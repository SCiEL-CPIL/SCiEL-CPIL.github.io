import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x, beta=50.0):
    """
    x: shape (N, 3)
    Returns energy and gradient.
    Energy: soft-min approximation of the minimum distance (pair and wall).
    We minimize -soft_min_dist, where soft_min uses log-sum-exp.
    """
    x = x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1.0)
    
    # Soft-min of pair distances (approximate min via log-sum-exp)
    # soft_min(d) = -1/beta * log(sum(exp(-beta*d)))
    # We work with distances, not squared distances
    d_pairs = np.sqrt(dist_sq_masked)
    d_pairs[~mask] = 1.0  # diagonal
    
    # For numerical stability, subtract max
    max_d = np.max(d_pairs)
    exp_terms = np.exp(-beta * (d_pairs - max_d))
    exp_terms[~mask] = 0.0  # zero out diagonal
    
    log_sum = np.log(np.sum(exp_terms))
    soft_min_pair = -1.0 / beta * (log_sum + beta * max_d)
    
    # Gradient of soft_min_pair w.r.t. x_i
    # d(soft_min)/dx_i = (1/beta) * sum_j w_j * (x_i - x_j) / |x_i - x_j|
    # where w_j = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
    
    weights = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)
    weights[~mask] = 0.0
    
    # Normalize diff by distance
    inv_dist = np.where(d_pairs > 1e-12, 1.0 / d_pairs, 0.0)
    inv_dist[~mask] = 0.0
    
    unit_diff = diff * inv_dist[:, :, np.newaxis]
    weighted_unit_diff = unit_diff * weights[:, :, np.newaxis]
    
    grad_soft_pair = (1.0 / beta) * np.sum(weighted_unit_diff, axis=1)
    
    # Wall distances
    d_left = x + 0.5
    d_right = 0.5 - x
    
    eps = 1e-10
    d_left = np.maximum(d_left, eps)
    d_right = np.maximum(d_right, eps)
    
    # Soft-min of wall distances
    all_wall_d = np.concatenate([d_left.flatten(), d_right.flatten()])
    max_wall = np.max(all_wall_d)
    exp_wall = np.exp(-beta * (all_wall_d - max_wall))
    log_sum_wall = np.log(np.sum(exp_wall))
    soft_min_wall = -1.0 / beta * (log_sum_wall + beta * max_wall)
    
    # Gradient of soft_min_wall w.r.t. x_i
    # d_left_k = x_ik + 0.5, d(x_ik)/d(x_ik) = 1
    # d_right_k = 0.5 - x_ik, d(x_ik)/d(x_ik) = -1
    w_left = np.exp(-beta * (d_left - max_wall)) / np.sum(exp_wall)
    w_right = np.exp(-beta * (d_right - max_wall)) / np.sum(exp_wall)
    
    grad_soft_wall = (1.0 / beta) * (w_left - w_right)
    
    # Total energy = -(soft_min_pair + soft_min_wall)
    # We want to maximize the minimum, so minimize the negative
    total_energy = -(soft_min_pair + soft_min_wall)
    total_grad = -(grad_soft_pair + grad_soft_wall)
    
    return total_energy, total_grad.flatten()

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _simulated_annealing(x0, start_temp=0.02, end_temp=0.001, n_iter=5000, beta=30.0):
    """
    Simulated annealing to escape local minima.
    """
    x = x0.copy()
    best_x = x0.copy()
    best_score = _calc_score(x0)
    
    np.random.seed(123)
    
    current_energy, _ = _energy_and_grad(x, beta=beta)
    
    for i in range(n_iter):
        # Annealing schedule
        t = i / n_iter
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Random perturbation
        dx = np.random.normal(0, temp, x.shape)
        x_new = x + dx
        x_new = np.clip(x_new, -0.499, 0.499)
        
        new_energy, _ = _energy_and_grad(x_new, beta=beta)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(temp, 1e-10)):
            x = x_new
            current_energy = new_energy
            
            # Track best by actual score
            if i % 50 == 0:
                score = _calc_score(x)
                if score > best_score:
                    best_score = score
                    best_x = x.copy()
    
    # Final score check
    score = _calc_score(x)
    if score > best_score:
        best_score = score
        best_x = x.copy()
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Base grid: 5x4x5 = 100 points
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    base_grid = grid[:N]
    
    best_x = base_grid.copy()
    best_score = _calc_score(best_x)
    
    # Objective function for scipy
    bounds = [(-0.5, 0.5)] * 3 * N
    
    # Strategy 1: Simulated annealing to explore
    # Reduced time for SA to allow more time for L-BFGS with increasing beta
    if time.time() < deadline - 60:
        sa_x, sa_score = _simulated_annealing(base_grid, start_temp=0.03, end_temp=0.001, n_iter=2000, beta=30.0)
        if sa_score > best_score:
            best_score = sa_score
            best_x = sa_x
    
    # Strategy 2: L-BFGS-B refinements with increasing beta (annealing the soft-min)
    # Start with low beta to find a good basin, then increase beta to sharpen the minimum
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    current_x = best_x.flatten()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def obj_beta(x_flat, beta=beta):
            e, g = _energy_and_grad(x_flat, beta=beta)
            return e, g
        
        try:
            # Increase maxiter for higher betas to ensure convergence
            max_iter = 100 if beta < 50 else 300
            res = minimize(obj_beta, current_x, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if res.success:
                new_x = res.x.reshape(N, 3)
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x
                
                # Use the result as the start for the next beta
                current_x = new_x.flatten()
                    
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish with very high beta
    if time.time() < deadline - 2:
        try:
            def obj_final(x_flat):
                e, g = _energy_and_grad(x_flat, beta=1000.0)
                return e, g
            
            res_final = minimize(obj_final, current_x, method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if res_final.success:
                final_x = res_final.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.clip(best_x, -0.5, 0.5)
    if not np.all(np.isfinite(best_x)):
        best_x = base_grid.copy()
    
    return best_x
