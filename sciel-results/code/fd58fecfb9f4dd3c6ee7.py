import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE = 115.0  # Safety margin

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min_objective(x, beta):
    """
    Objective function to minimize: -soft_min(distances)
    which is equivalent to soft_max(-distances) or just minimizing the log-sum-exp of -distances.
    We want to maximize the minimum distance.
    soft_min(D, beta) = -1/beta * log(sum(exp(-beta * d)))
    We minimize J = 1/beta * log(sum(exp(-beta * d)))
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    # dist to wall for coordinate k is 0.5 - |c_k|
    wall_dists = HALF - np.abs(c)
    
    # Combine all distances into one array
    # We need to be careful with memory. 100*99/2 pairs + 100*3 walls = 4950 + 300 = 5250 distances.
    # This is small enough to handle directly.
    
    # For numerical stability, use the max trick for log-sum-exp
    # log(sum(exp(-beta * d))) = -beta * max(d) + log(sum(exp(-beta * (d - max_d))))
    # Wait, we are minimizing 1/beta * log(sum(exp(-beta * d))).
    # Let M = min(d). Then d = M + (d-M).
    # sum(exp(-beta * (M + d - M))) = exp(-beta * M) * sum(exp(-beta * (d - M)))
    # log(...) = -beta * M + log(sum(exp(-beta * (d - M))))
    # J = M - 1/beta * log(sum(exp(-beta * (d - M))))
    # As beta -> inf, J -> M.
    
    # Let's just compute the raw sum of exp(-beta * d) with scaling.
    # To avoid overflow/underflow, subtract the minimum distance from all distances before exponentiation.
    all_dists = np.concatenate([dists[~np.eye(N, dtype=bool)].flatten(), wall_dists.flatten()])
    
    min_d = np.min(all_dists)
    scaled_dists = all_dists - min_d
    
    # exp(-beta * scaled_dists)
    # scaled_dists >= 0, so exp(-beta * scaled_dists) <= 1. No overflow.
    # Underflow is possible for large beta * scaled_dists, but that's fine (contribution becomes 0).
    
    exp_terms = np.exp(-beta * scaled_dists)
    sum_exp = np.sum(exp_terms)
    
    # J = min_d - (1/beta) * log(sum_exp)
    # We want to MINIMIZE J? 
    # If we minimize J, we are minimizing the soft-min.
    # But we want to MAXIMIZE the min distance.
    # So we should MINIMIZE -soft_min = - (min_d - 1/beta * log(sum_exp)) = -min_d + 1/beta * log(sum_exp).
    # Let's check the gradient direction.
    # If we increase d, soft_min increases. So we want to increase d.
    # So we should minimize -soft_min.
    
    val = -min_d + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # d/dx [ -min_d + 1/beta * log(sum(exp(-beta*(d-min_d)))) ]
    # Note: min_d depends on x, but in the limit beta->inf, the derivative of min_d is the subgradient of the min.
    # The soft-min gradient is a weighted average of the gradients of the individual distances.
    # grad(soft_min) = sum( p_i * grad(d_i) ) where p_i = exp(-beta*d_i)/sum(exp(-beta*d_i))
    # grad(J) = - grad(soft_min)
    
    # We need to compute the weights p_i.
    # p_i = exp(-beta * (d_i - min_d)) / sum_exp
    
    weights = exp_terms / sum_exp
    
    # Now we need to sum weights * grad(d_i) for all pairs and walls.
    # This is expensive to do explicitly in Python loops.
    # We can reconstruct the gradients using the weights.
    
    # Reshape weights back
    # First N*(N-1)/2 are pairs, last N*3 are walls.
    num_pairs = N * (N - 1) // 2
    pair_weights = weights[:num_pairs]
    wall_weights = weights[num_pairs:]
    
    # Pair gradients
    # We have a flat array of pair weights. We need to map them back to (i, j) indices.
    # This is tricky to vectorize perfectly without creating a huge matrix, but N=100 is small.
    # Let's create the full matrix of weights for pairs.
    w_mat = np.zeros((N, N))
    # Fill upper triangle
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w_mat[i, j] = pair_weights[idx]
            w_mat[j, i] = pair_weights[idx] # Symmetric
            idx += 1
            
    # grad_pair_i = - sum_j w_mat[i,j] * (c_i - c_j) / d_ij
    # We have diff matrix and dists matrix.
    # unit_vecs = diff / dists (with 0 on diagonal)
    # We need to handle d=0.
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # grad_pair = - sum_j w_mat[i,j] * unit_vecs[i,j]
    # Note: The sign.
    # We are minimizing -soft_min.
    # grad(-soft_min) = - grad(soft_min) = - sum(p_i grad(d_i)).
    # So grad_pair_component = - np.sum(w_mat[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad_pair = -np.sum(w_mat[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall gradients
    # wall_dists shape (N, 3). wall_weights shape (N, 3).
    # grad(wall_dist_ik) = -sign(c_ik)
    # grad(-soft_min)_wall = - sum_k p_wall_ik * (-sign(c_ik)) = sum_k p_wall_ik * sign(c_ik)
    signs = np.sign(c)
    grad_wall = np.sum(wall_weights[:, :, np.newaxis] * signs, axis=1)
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    return val, total_grad

def _optimize_step(init_pts, beta, time_limit):
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_objective(x, beta)
        
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 1000, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        return result.x.reshape(N, 3)
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def construct():
    t0 = time.time()
    
    # Initialization: 5x5x4 grid
    # 5 in x, 5 in y, 4 in z
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    
    grid_pts = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    
    # Add small noise to break symmetry
    rng = np.random.default_rng(42)
    grid_pts = grid_pts + rng.normal(0, 0.01, size=(N, 3))
    grid_pts = np.clip(grid_pts, -0.49, 0.49)
    
    best_pts = grid_pts
    best_r = _compute_radius(best_pts)
    
    # Annealing schedule for beta
    betas = [10, 20, 50, 100, 200, 500]
    
    current_pts = best_pts.copy()
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Try a few restarts for each beta if time permits?
        # For now, one run per beta, but with noise added to current best.
        
        # Add noise to help escape local minima
        noise = rng.normal(0, 0.02, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining_time = DEADLINE - (time.time() - t0) - 5.0
        if remaining_time < 2.0:
            break
            
        opt_pts = _optimize_step(init_pts, beta, time_limit=remaining_time)
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # If no improvement, keep current_pts as base for next iteration
                # But maybe the optimizer moved it to a better basin for higher beta?
                # Let's check if the new points are "better" in terms of energy for current beta?
                # No, just stick to best radius for stability.
                pass

    # Final Polish: Use the best configuration and try to push radius further
    # using a penalty method with a high beta or direct gradient ascent on min distance.
    # We can run _optimize_step with a very high beta on the best_pts.
    
    if time.time() - t0 < DEADLINE - 2.0:
        # High beta polish
        high_betas = [1000, 5000]
        for beta in high_betas:
            if time.time() - t0 > DEADLINE - 1.0:
                break
            remaining_time = DEADLINE - (time.time() - t0) - 0.5
            if remaining_time < 1.0:
                break
            
            # Small noise
            noise = rng.normal(0, 0.005, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            opt_pts = _optimize_step(init_pts, beta, time_limit=remaining_time)
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    return best_pts
