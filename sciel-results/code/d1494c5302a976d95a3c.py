"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the radius of the packing."""
    # Center to boundary distance
    # centers shape (N, 3), bounds [-0.5, 0.5]
    # dist to boundary = 0.5 - abs(c)
    # min dist to boundary
    dist_to_bound = np.min(0.5 - np.abs(centers), axis=1)
    r_bound = np.min(dist_to_bound)
    
    # Pairwise distances
    # To save memory, we can compute in chunks or use a trick
    # But N=100 is small.
    # c1 = centers[:, None, :], c2 = centers[None, :, :]
    # diff = c1 - c2
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # r_pair = np.min(dists) / 2
    
    # More efficient for N=100:
    # Compute squared distances
    # d_sq = np.sum(centers**2, axis=1)
    # d_sq = d_sq[:, None] + d_sq[None, :] - 2 * centers @ centers.T
    # np.fill_diagonal(d_sq, np.inf)
    # d_sq = np.maximum(d_sq, 0)
    # dists = np.sqrt(d_sq)
    # r_pair = np.min(dists) / 2
    
    # Let's use the vectorized way
    c1 = centers[:, None, :]
    c2 = centers[None, :, :]
    diff = c1 - c2
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2
    
    return min(r_bound, r_pair)

def _lse_objective(params, beta):
    """
    Objective function to minimize.
    params: flat array of shape (3*N,)
    beta: temperature parameter for LogSumExp approximation of min distance.
    Returns: (value, gradient)
    """
    centers = params.reshape(-1, 3)
    
    # Compute pairwise distances
    c1 = centers[:, None, :]
    c2 = centers[None, :, :]
    diff = c1 - c2
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, np.inf)
    
    # LogSumExp approximation of min distance
    # min(d) approx -1/beta * log(sum(exp(-beta * d)))
    # We want to MAXIMIZE min(d), so we MINIMIZE -min(d)
    # So we minimize 1/beta * log(sum(exp(-beta * d)))
    # Let S = sum(exp(-beta * d))
    # f = 1/beta * log(S)
    # grad f = 1/beta * (1/S) * sum(exp(-beta * d) * grad(-beta * d))
    #        = - sum(exp(-beta * d) * grad(d)) / S
    
    # To avoid overflow/underflow, use log-sum-exp trick
    # log(S) = max_d + log(sum(exp(-beta * (d - max_d))))
    # But here we have exp(-beta * d).
    # Let u = -beta * dists_masked
    # We want log(sum(exp(u)))
    u = -beta * dists_masked
    u_max = np.max(u)
    if np.isinf(u_max):
        # Should not happen if dists are finite
        return 0, np.zeros_like(params)
    
    log_S = u_max + np.log(np.sum(np.exp(u - u_max)))
    value = log_S / beta
    
    # Gradient
    # weights = exp(u - u_max) / sum(exp(u - u_max))
    # This is softmax(u)
    exp_u = np.exp(u - u_max)
    sum_exp_u = np.sum(exp_u)
    weights = exp_u / sum_exp_u
    
    # grad d_ij wrt centers
    # d_ij = ||c_i - c_j||
    # grad d_ij = (c_i - c_j) / d_ij
    # Sum over j for fixed i: sum_j weights_ij * (c_i - c_j)/d_ij
    # Note: weights is symmetric.
    
    # grad_params shape (3*N,)
    grad_centers = np.zeros_like(centers)
    
    # Vectorized gradient calculation
    # diff is (N, N, 3)
    # dists is (N, N)
    # weights is (N, N)
    
    # We need sum_j weights[i,j] * (c_i - c_j) / d_ij
    # Let G[i,j] = weights[i,j] / d_ij * (c_i - c_j)
    # But d_ij is in dists.
    # Avoid division by zero (diagonal is inf, so weight 0, but 0/inf is 0).
    # Safe divide
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = 1.0 / dists
        inv_dists[~mask] = 0.0 # Diagonal
        
    # factor[i,j] = weights[i,j] * inv_dists[i,j]
    factor = weights * inv_dists
    
    # grad_centers[i] = sum_j factor[i,j] * (c_i - c_j)
    # = sum_j factor[i,j] * c_i - sum_j factor[i,j] * c_j
    # = c_i * sum_j factor[i,j] - sum_j factor[i,j] * c_j
    
    row_sums = np.sum(factor, axis=1)
    # shape (N,)
    
    # sum_j factor[i,j] * c_j
    # factor is (N,N), c is (N,3)
    weighted_sum = factor @ centers # shape (N,3)
    
    grad_centers = row_sums[:, None] * centers - weighted_sum
    
    # The objective was log_S / beta
    # grad objective = 1/beta * grad log_S
    # grad log_S = grad (u_max + log(sum(exp(u-u_max))))
    # This is tricky to derive fully vectorized with u_max.
    # However, we can use the property that grad log(sum(exp(u))) = softmax(u) * grad u
    # u = -beta * dists
    # grad u = -beta * grad dists
    # So grad log_S = softmax(u) * (-beta * grad dists)
    # Wait, the u_max shift cancels in the gradient of the log-sum-exp part?
    # d/dx log(sum(exp(u(x)))) = (sum(exp(u) * u')) / sum(exp(u)) = softmax(u) * u'
    # Yes.
    # So grad log_S = sum_i softmax(u)_i * u'_i
    # Here u is the vector of all -beta * d_ij.
    # So grad log_S = sum_{i,j} weights_ij * (-beta * grad d_ij)
    # = -beta * sum_{i,j} weights_ij * grad d_ij
    
    # So grad value = 1/beta * (-beta * sum ... ) = - sum ...
    # My previous derivation:
    # grad f = - sum(weights * grad d) / S ... wait.
    # f = 1/beta * log S
    # grad f = 1/beta * (1/S) * sum(exp(-beta d) * (-beta grad d))
    # = - 1/S * sum(exp(-beta d) * grad d)
    # = - sum( (exp(-beta d)/S) * grad d )
    # = - sum( weights * grad d )
    
    # So grad_centers = - sum_{j} weights[i,j] * (c_i - c_j)/d_ij
    # My calculation above was:
    # grad_centers = row_sums * centers - weighted_sum
    # This is sum_j factor[i,j] * (c_i - c_j) where factor = weights * inv_dists
    # So grad_centers calculated above is sum(weights * grad d)
    # Therefore, grad value = - grad_centers
    
    grad_params = -grad_centers.flatten()
    
    return value, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial Grid: 4x5x5
    # x: 4 points, y: 5 points, z: 5 points
    x_coords = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)
    y_coords = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    z_coords = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    
    best_centers = grid.copy()
    best_radius = _compute_radius(best_centers)
    
    # Add small jitter
    np.random.seed(42)
    jitter = np.random.normal(0, 0.01, grid.shape)
    # Ensure within bounds
    initial_centers = np.clip(grid + jitter, -0.49, 0.49)
    
    betas = [5, 20, 100, 500]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        res = minimize(
            _lse_objective, 
            initial_centers.flatten(), 
            args=(beta,), 
            method='L-BFGS-B', 
            bounds=[(-0.49, 0.49)] * (3 * N),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success:
            new_centers = res.x.reshape(-1, 3)
            # Clip to be safe
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers
                initial_centers = new_centers # Use best as start for next beta
    
    return best_centers

# EVOLVE-BLOCK-END
