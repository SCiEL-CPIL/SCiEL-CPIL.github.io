import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_score(centres):
    """
    Compute the maximum radius for the given centres.
    """
    # Pairwise distances
    # centres shape (N, 3)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Cube is [-0.5, 0.5]^3
    # Distance to wall is 0.5 - |c_i|
    wall_dists = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def soft_min_objective_and_grad(x, beta):
    """
    x: flattened array of shape (3*N,)
    beta: annealing parameter
    Returns: objective value, gradient
    """
    c = x.reshape(N, 3)
    
    # Calculate pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = dists[mask]
    # Gradients for pairs
    # grad_dists[i, j] = (c_i - c_j) / dists[i, j]
    # We need to accumulate gradient for c_i and c_j
    # Let's compute the weights first
    # To avoid underflow/overflow in exp(-beta * dists)
    # Use log-sum-exp trick for the sum part, but we need the gradient which is a weighted average
    
    # Weights w_k = exp(-beta * d_k) / sum(exp(-beta * d_k))
    # Objective F = (1/beta) * log(sum(exp(-beta * d_k)))
    # Grad F = - sum(w_k * grad_d_k)
    
    # Wall distances
    # d_w = 0.5 - |c_i|
    # grad_d_w = -sign(c_i)
    
    # Combine all distances into one array for log-sum-exp
    # d_pairs (N*(N-1)/2), d_walls (3*N)
    
    # Efficiently compute exp(-beta * d)
    # d_pairs can be 0 if points coincide, leading to inf.
    # Add a small epsilon to distances to avoid division by zero in gradient
    eps = 1e-12
    dists_safe = np.maximum(dists_masked, eps)
    wall_dists_raw = 0.5 - np.abs(c)
    wall_dists_safe = np.maximum(wall_dists_raw, eps)
    
    all_dists = np.concatenate([dists_safe, wall_dists_safe])
    
    # Log-sum-exp for stability
    # m = max(-beta * all_dists)
    neg_beta_d = -beta * all_dists
    m = np.max(neg_beta_d)
    # If m is very negative, exp(neg_beta_d - m) might underflow, but that's ok if m is handled.
    # Actually, if all d are large, neg_beta_d is very negative.
    # Let's just compute exp(neg_beta_d - m)
    exp_terms = np.exp(neg_beta_d - m)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Fallback, shouldn't happen
        val = -np.inf
        grad = np.zeros_like(x)
        return val, grad
    
    obj_val = (1.0 / beta) * (m + np.log(sum_exp))
    
    # Weights
    # w_k = exp(-beta * d_k) / sum_exp
    # Note: exp(-beta * d_k) = exp(neg_beta_d_k) = exp(neg_beta_d_k - m) * exp(m)
    # So w_k = exp_terms_k * exp(m) / sum_exp
    # The exp(m) cancels out in the weighted sum if we are careful, or we just compute weights directly.
    # w_k = exp_terms_k / sum_exp * exp(m) ... wait.
    # Sum w_k = 1.
    # w_k = exp(neg_beta_d_k) / Sum(exp(neg_beta_d_j))
    # w_k = (exp(neg_beta_d_k - m) * exp(m)) / (Sum(exp(neg_beta_d_j - m)) * exp(m))
    # w_k = exp_terms_k / sum_exp
    weights = exp_terms / sum_exp
    
    # Gradient calculation
    # Grad F = - sum_k w_k * grad_d_k
    # grad_d_k for pairs:
    # For pair (i, j), dist = ||c_i - c_j||
    # grad_{c_i} dist = (c_i - c_j) / dist
    # grad_{c_j} dist = (c_j - c_i) / dist
    
    # We need to map the flat index of all_dists back to the pairs and walls.
    # First half: pairs. Second half: walls.
    
    num_pairs = len(dists_safe)
    num_walls = len(wall_dists_safe)
    
    w_pairs = weights[:num_pairs]
    w_walls = weights[num_pairs:]
    
    grad_c = np.zeros_like(c)
    
    # Pairs
    # Reconstruct the (N, N) matrix of weights for pairs?
    # Or iterate? N=50, pairs=1225. Iteration in Python is slow.
    # Vectorized approach:
    # Create a matrix W_pairs of shape (N, N) with weights.
    # W_pairs[i, j] = w_pairs[idx]
    # We need to scatter the flat array back to (N, N).
    
    # Indices for pairs
    ii, jj = np.triu_indices(N, 1)
    W_pairs_mat = np.zeros((N, N))
    W_pairs_mat[ii, jj] = w_pairs
    W_pairs_mat[jj, ii] = w_pairs
    
    # Gradient contribution from pairs
    # grad_c_i += sum_j W_pairs_mat[i, j] * (c_i - c_j) / dists[i, j]
    # grad_c_j += sum_i W_pairs_mat[i, j] * (c_j - c_i) / dists[i, j]
    
    # Let's compute the force vectors
    # F_ij = W_pairs_mat[i, j] * (c_i - c_j) / dists[i, j]
    # This is the gradient of dist wrt c_i, multiplied by weight.
    # We want - sum w_k grad_d_k.
    # So we subtract the sum of these forces from c_i?
    # grad_obj = - sum w_k grad_d_k
    
    # Vectorized force calculation:
    # diff is (N, N, 3)
    # dists_safe is (N, N)
    # Unit vectors: diff / dists_safe[..., np.newaxis]
    # Mask diagonal to avoid 0/0
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / np.maximum(dists_safe, eps)[..., np.newaxis]
    
    # Set diagonal to 0
    unit_vecs = np.where(mask[..., np.newaxis], unit_vecs, 0)
    
    # Force on i from j: W[i, j] * unit_vec[i, j]
    # Force on j from i: W[i, j] * unit_vec[j, i] = W[i, j] * (-unit_vec[i, j])
    
    # Sum over j for each i:
    # grad_c_i = - sum_j W[i, j] * unit_vec[i, j]
    
    pair_grad = -np.sum(W_pairs_mat[..., np.newaxis] * unit_vecs, axis=1)
    
    # Walls
    # d_w = 0.5 - |c_i|
    # grad_d_w = -sign(c_i)
    # obj_grad = - sum w_walls * grad_d_w = - sum w_walls * (-sign(c_i)) = sum w_walls * sign(c_i)
    
    # wall_dists_safe is flattened (N, 3)
    # w_walls is flat (3*N)
    # Reshape w_walls to (N, 3)
    W_walls_mat = w_walls.reshape(N, 3)
    
    # sign(c)
    signs = np.sign(c)
    
    wall_grad = W_walls_mat * signs
    
    total_grad = pair_grad + wall_grad
    
    # Flatten gradient
    grad_flat = total_grad.flatten()
    
    return obj_val, grad_flat

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_score = -np.inf
    best_centres = None
    
    # Initializations
    # 1. Random
    # 2. Grid-like
    inits = []
    
    # Random inits
    for i in range(5):
        # Uniform in [-0.4, 0.4] to start safely inside
        c = np.random.uniform(-0.4, 0.4, (N, 3))
        inits.append(c)
        
    # Grid init
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add some noise
    grid = grid[:N] + np.random.normal(0, 0.01, (N, 3))
    # Clip to bounds
    grid = np.clip(grid, -0.49, 0.49)
    inits.append(grid)
    
    # Annealing schedule for beta
    # Start low, go high
    betas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for init_c in inits:
        if time.time() - start_time > time_limit:
            break
            
        current_c = init_c.copy()
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            x0 = current_c.flatten()
            
            # L-BFGS-B
            # Bounds: [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (3 * N)
            
            try:
                res = minimize(
                    soft_min_objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_c = res.x.reshape(N, 3)
            except Exception as e:
                # If optimization fails, keep previous
                pass
            
            # Evaluate true score periodically or at the end of each beta step?
            # Evaluating true score is cheap (N^2).
            score = compute_score(current_c)
            if score > best_score:
                best_score = score
                best_centres = current_c.copy()
                
    # Final polish with very high beta or direct local search
    # If we have time, do a few more iterations with high beta on the best solution
    if best_centres is not None and time.time() - start_time < time_limit - 5:
        current_c = best_centres.copy()
        for beta in [1000.0, 2000.0]:
            if time.time() - start_time > time_limit - 5:
                break
            x0 = current_c.flatten()
            bounds = [(-0.5, 0.5)] * (3 * N)
            try:
                res = minimize(
                    soft_min_objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_c = res.x.reshape(N, 3)
            except:
                pass
            score = compute_score(current_c)
            if score > best_score:
                best_score = score
                best_centres = current_c.copy()
                
    # Ensure validity
    if best_centres is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
        
    # Final check: ensure inside bounds
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres

# EVOLVE-BLOCK-END
