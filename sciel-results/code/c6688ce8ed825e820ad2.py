"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Minimum distance to boundary
    min_boundary = 0.5 - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_boundary, min_pair)


def _softmin_energy(centers, beta=15.0):
    """
    Compute a smooth soft-min energy and its gradient.
    The energy is -soft_min(distances) where soft_min approximates the minimum distance.
    We want to maximize the minimum distance, so we minimize this energy.
    
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    Energy = -soft_min = 1/beta * log(sum(exp(-beta * x_i)))
    
    x_i includes all pairwise distances and boundary distances.
    """
    n = len(centers)
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)  # shape (n, 3)
    
    # For numerical stability, use log-sum-exp trick
    # Z = sum(exp(-beta * d_i))
    # log(Z) = max(-beta*d) + log(sum(exp(-beta*d - max(-beta*d))))
    
    # Collect all distances
    # Pairwise: only upper triangle to avoid double counting
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask_upper]
    
    # Boundary: all n*3 values
    bound_dists = boundary_dists.flatten()
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Handle potential non-positive distances (shouldn't happen if clipped)
    all_dists = np.maximum(all_dists, 1e-10)
    
    # Log-sum-exp for numerical stability
    neg_beta_d = -beta * all_dists
    max_val = np.max(neg_beta_d)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_d - max_val)))
    
    energy = log_sum_exp / beta
    
    # Gradient computation
    # dE/dc = 1/beta * 1/Z * dZ/dc
    # Z = sum(exp(-beta * d_i))
    # dZ/dc_j = sum_i exp(-beta * d_i) * (-beta) * dd_i/dc_j
    
    # For pairwise distance d_ij = |c_i - c_j|
    # dd_ij/dc_i = (c_i - c_j) / d_ij
    # dd_ij/dc_j = (c_j - c_i) / d_ij
    
    # For boundary distance d_ik = 0.5 - |c_ik|
    # dd_ik/dc_ik = -sign(c_ik)
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # weights_ij = exp(-beta * d_ij) / Z
    # But we can compute it more efficiently
    
    # Let's compute the unnormalized weights
    pair_weights = np.exp(-beta * dists)
    np.fill_diagonal(pair_weights, 0.0)
    
    # Normalize by Z_pair = sum of pair_weights (upper triangle only, but we have full matrix)
    # Actually, let's just use the full matrix for simplicity, but divide by 2 since we counted each pair twice
    Z_pair = np.sum(pair_weights) / 2.0  # Each pair counted twice
    Z_bound = np.sum(np.exp(-beta * boundary_dists))
    Z_total = Z_pair + Z_bound
    
    # Pairwise contribution to gradient
    # grad_i += sum_j (pair_weights[i,j] / Z_total) * (-beta) * (c_i - c_j) / d_ij
    # Let's compute this as: for each i, sum over j of weight * (c_i - c_j) / d_ij
    
    # Create a matrix of unit vectors scaled by 1/d
    # unit_vec[i,j] = (c_i - c_j) / d_ij
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diffs / dists[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs[:, :, 0], 0.0)  # This won't work properly, let's do it differently
    
    # Recompute more carefully
    # For each pair (i,j), the contribution to grad[i] is:
    # (pair_weights[i,j] / Z_total) * (-beta) * (c_i - c_j) / d_ij
    # And to grad[j] is:
    # (pair_weights[j,i] / Z_total) * (-beta) * (c_j - c_i) / d_ij
    
    # Let's compute a matrix W where W[i,j] = pair_weights[i,j] / (Z_total * d_ij)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_weights / (Z_total * dists)
    np.fill_diagonal(W, 0.0)
    
    # grad_pair[i] = -beta * sum_j W[i,j] * (c_i - c_j)
    #              = -beta * (c_i * sum_j W[i,j] - sum_j W[i,j] * c_j)
    
    sum_W = np.sum(W, axis=1)
    W_c = W @ centers
    
    grad_pair = -beta * (centers * sum_W[:, np.newaxis] - W_c)
    
    # Boundary gradient
    # For boundary distance d_ik = 0.5 - |c_ik|
    # Contribution to grad[i,k] = (exp(-beta * d_ik) / Z_total) * (-beta) * (-sign(c_ik))
    #                           = beta * sign(c_ik) * exp(-beta * d_ik) / Z_total
    
    signs = np.sign(centers)
    # Handle zero: use 0
    signs[centers == 0] = 0.0
    
    bound_weights = np.exp(-beta * boundary_dists) / Z_total
    grad_boundary = beta * signs * bound_weights
    
    grad = grad_pair + grad_boundary
    
    return energy, grad


def _optimize_softmin(centers, time_budget=10.0, beta_start=10.0, beta_end=25.0):
    """
    Use L-BFGS-B to minimize the soft-min energy.
    Adaptive beta: start low for smooth landscape, increase for sharper minimum.
    """
    start_time = time.time()
    
    # Two-phase optimization:
    # Phase 1: Low beta (smooth)
    # Phase 2: High beta (sharp)
    
    def run_optimization(c_init, beta, time_limit):
        def obj(x):
            c = x.reshape(-1, 3)
            c = np.clip(c, -0.5, 0.5)
            e, g = _softmin_energy(c, beta)
            return e, g.flatten()
        
        x0 = c_init.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def callback(xk):
            if time.time() - start_time > time_limit:
                return True
            return False
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'callback': callback, 'disp': False})
            new_centers = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
        except Exception:
            new_centers = c_init.copy()
        
        return new_centers
    
    # Phase 1: Smooth optimization
    time_phase1 = min(time_budget * 0.4, time_budget - 1.0)
    if time_phase1 > 0.5:
        c_phase1 = run_optimization(centers, beta_start, time_phase1)
    else:
        c_phase1 = centers.copy()
    
    # Phase 2: Sharp optimization
    time_phase2 = min(time_budget * 0.6, time_budget - time.time() + start_time - start_time)
    time_phase2 = min(time_budget * 0.6, max(0, time_budget - (time.time() - start_time)))
    if time_phase2 > 0.5:
        c_final = run_optimization(c_phase1, beta_end, time_phase2)
    else:
        c_final = c_phase1
    
    return c_final


def _polish_efficient(centers, time_budget=10.0):
    """
    Efficient local polish: only check distances to nearby points.
    """
    start_time = time.time()
    polish_centers = centers.copy()
    polish_radius = _compute_radius(polish_centers)
    
    T = 0.003
    n_steps = int(min(10000, time_budget * 50))
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget - 0.5:
            break
        
        idx = np.random.randint(N)
        delta = np.random.normal(0, T, 3)
        new_centers = polish_centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Quick radius check
        new_radius = _compute_radius(new_centers)
        if new_radius > polish_radius:
            polish_centers = new_centers
            polish_radius = new_radius
            T *= 0.98
        else:
            T *= 0.99
    
    return polish_centers, polish_radius


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin

    np.random.seed(42)

    # Start with a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()

    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Multiple restarts with soft-min optimization
    n_restarts = 0
    while time.time() < deadline - 20.0:  # Leave 20s for final polish
        # Generate a new random starting point
        if n_restarts % 3 == 0:
            # Perturbed best
            c_init = best_centers.copy()
            c_init += np.random.normal(0, 0.03, c_init.shape)
        elif n_restarts % 3 == 1:
            # Random points
            c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        else:
            # Perturbed grid
            c_init = grid[:N].copy()
            c_init += np.random.normal(0, 0.04, c_init.shape)
        
        c_init = np.clip(c_init, -0.5, 0.5)
        
        # Optimize
        time_for_opt = min(12.0, deadline - 20.0 - time.time())
        if time_for_opt < 2.0:
            break
            
        c_opt = _optimize_softmin(c_init, time_budget=time_for_opt)
        
        r_opt = _compute_radius(c_opt)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
        
        n_restarts += 1

    # Final polish with efficient local moves
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        polish_time = min(remaining_time - 1.0, 15.0)
        polish_centers, polish_radius = _polish_efficient(best_centers, time_budget=polish_time)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers.copy()

    return best_centers
# EVOLVE-BLOCK-END
