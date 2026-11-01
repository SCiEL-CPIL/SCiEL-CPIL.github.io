import numpy as np
from scipy.optimize import minimize

N = 50

def soft_min_obj(c, beta):
    """
    Objective function to minimize.
    c: (N, 3) array of centers.
    beta: inverse temperature for soft-min.
    """
    # Flatten c to 1D for scipy
    # Reshape to (N, 3)
    pts = c.reshape(N, 3)
    
    # Pairwise distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    
    # Soft min of distances
    # We want to maximize min dist, so minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow/underflow, we can shift
    # d_min is roughly 0.1, beta 20 -> exp(-2) ~ 0.13
    # Sum of 1225 terms.
    # log(sum(exp(-beta*d)))
    
    # Numerical stability
    min_d = np.min(dists)
    shifted = dists - min_d
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) - beta * min_d
    soft_min_val = -1.0 / beta * log_sum
    
    return -soft_min_val

def soft_min_grad(c, beta):
    pts = c.reshape(N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dists_flat = dists[mask]
    
    min_d = np.min(dists_flat)
    shifted = dists_flat - min_d
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    # Gradient of -soft_min w.r.t c
    # soft_min = -1/beta log(sum exp(-beta d))
    # d(soft_min)/d c_i = -1/beta * (1/sum_exp) * sum_j d(sum_exp)/d c_i
    # d(sum_exp)/d c_i = sum_j exp(-beta d_ij) * d(-beta d_ij)/d c_i
    # d(d_ij)/d c_i = (c_i - c_j) / d_ij
    
    # This is O(N^2).
    # Let's compute the gradient vector.
    
    # We need d(soft_min)/d c_i
    # Let S = sum_{j != i} exp(-beta d_ij)
    # d(soft_min)/d c_i = -1/beta * (1/S_total) * sum_{j != i} exp(-beta d_ij) * (-beta) * (c_i - c_j)/d_ij
    # = (1/S_total) * sum_{j != i} exp(-beta d_ij) * (c_i - c_j)/d_ij
    
    # S_total is the same for all i in the sum? No, S_total is global sum.
    # Wait, the objective is f(c) = -1/beta log( sum_{i<j} exp(-beta d_ij) )
    # Let K = sum_{i<j} exp(-beta d_ij)
    # df/dc_i = -1/beta * (1/K) * dK/dc_i
    # dK/dc_i = sum_{j != i} exp(-beta d_ij) * (-beta) * (c_i - c_j)/d_ij
    # Note: for j < i, the term is in K. For j > i, it is in K.
    # The derivative of d_ij w.r.t c_i is (c_i - c_j)/d_ij.
    # The derivative of d_ji w.r.t c_i is (c_j - c_i)/d_ji = -(c_i - c_j)/d_ij.
    # So dK/dc_i = sum_{j != i} exp(-beta d_ij) * (-beta) * (c_i - c_j)/d_ij
    # Note that d_ij = d_ji.
    
    # So df/dc_i = -1/beta * (1/K) * (-beta) * sum_{j != i} exp(-beta d_ij) * (c_i - c_j)/d_ij
    # = (1/K) * sum_{j != i} exp(-beta d_ij) * (c_i - c_j)/d_ij
    
    # We need to compute this for all i.
    
    # Vectorized approach:
    # We have dists (N, N) and diff (N, N, 3)
    # We need weights w_ij = exp(-beta * dists_ij) / K
    # Then grad_i = sum_{j != i} w_ij * (c_i - c_j) / dists_ij
    
    # Handle dists=0 (should not happen if initialized well, but just in case)
    dists_safe = np.where(dists > 1e-8, dists, 1e-8)
    
    # Weights
    # We need K = sum_{i<j} exp(-beta d_ij)
    # K = 0.5 * sum_{i != j} exp(-beta d_ij)
    sum_exp_all = np.sum(exp_terms) # This is sum over i<j
    K = sum_exp_all
    
    # We need the matrix of weights W where W[i,j] = exp(-beta d_ij) / K for i != j
    # But exp_terms was computed on flattened i<j.
    # Let's reconstruct the full matrix of exp terms.
    exp_full = np.exp(-beta * dists)
    np.fill_diagonal(exp_full, 0)
    
    # The sum in K is half of the sum of exp_full.
    # K = 0.5 * np.sum(exp_full)
    # Let's verify: sum_{i<j} is half of sum_{i!=j}. Yes.
    # So we can just use exp_full.
    
    # grad_i = (1/K) * sum_{j != i} exp(-beta d_ij) * (c_i - c_j) / d_ij
    # Let's compute the term T_i = sum_{j != i} exp(-beta d_ij) * (c_i - c_j) / d_ij
    
    # (c_i - c_j) is diff[i, j]
    # We need to divide diff by dists.
    # diff / dists[..., np.newaxis]
    unit_vecs = diff / dists_safe[..., np.newaxis]
    np.fill_diagonal(unit_vecs, 0) # Avoid div by 0 on diag
    
    # Weighted sum
    # G[i] = sum_j exp_full[i, j] * unit_vecs[i, j]
    G = np.sum(exp_full[..., np.newaxis] * unit_vecs, axis=1)
    
    # Normalize by K
    # K = 0.5 * sum(exp_full)
    K_val = 0.5 * np.sum(exp_full)
    if K_val < 1e-10:
        return np.zeros_like(c)
        
    grad = G / K_val
    
    return grad.flatten()

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_r = 0.0
    best_c = None
    
    # Use a grid initialization as a strong start
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Add some random restarts
    restarts = 5
    
    for i in range(restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            c_init = c0
        else:
            c_init = np.random.uniform(-0.45, 0.45, size=(N, 3))
            
        # Optimize
        try:
            res = minimize(
                soft_min_obj, 
                c_init.flatten(), 
                method='L-BFGS-B', 
                jac=soft_min_grad,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
        except Exception:
            continue
            
        c_opt = res.x.reshape(N, 3)
        
        # Calculate radius
        # Min pair distance
        diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        min_pair_dist = np.min(dists[mask])
        
        # Min distance to boundary
        max_coord = np.max(np.abs(c_opt), axis=1)
        min_bound_dist = 0.5 - np.min(max_coord)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        
        if r > best_r:
            best_r = r
            best_c = c_opt
            
    if best_c is None:
        return c0
        
    return best_c
