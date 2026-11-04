"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the actual radius (minimum of half-pairwise-distances and wall-distances)."""
    if len(centers) == 0:
        return 0.0
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pd = pdist(centers)
    min_pair = np.min(pd) / 2.0 if len(pd) > 0 else 0.0
    return min(min_wall, min_pair)

def _soft_min_obj_grad(centers, tau):
    """
    Compute soft-min of all relevant distances and its gradient w.r.t. centers.
    Returns (obj_value, grad) where obj_value is the soft-min (to be maximized).
    """
    n = len(centers)
    # Collect all distances and their gradients
    # We'll compute the soft-min of: all pairwise distances/2 and all wall distances
    # Use log-sum-exp for numerical stability
    
    # Wall distances: d = 0.5 - |c_ik| for each component
    # Pairwise: d = |c_i - c_j| / 2
    
    # For efficiency, compute in chunks
    log_sum = 0.0
    # We need to track gradients, so let's build them
    
    # Initialize gradient accumulator
    grad = np.zeros_like(centers)
    
    # Wall contributions
    # d_wall_ik = 0.5 - |c_ik|
    # exp(-d/tau) for each
    abs_c = np.abs(centers)
    d_walls = 0.5 - abs_c
    # Clip to avoid issues
    d_walls_clipped = np.clip(d_walls, 1e-10, None)
    exp_walls = np.exp(-d_walls_clipped / tau)
    
    # Pairwise contributions
    # Use scipy for pairwise distances
    from scipy.spatial.distance import pdist, squareform
    pd = pdist(centers)
    d_pairs = pd / 2.0
    d_pairs_clipped = np.clip(d_pairs, 1e-10, None)
    exp_pairs = np.exp(-d_pairs_clipped / tau)
    
    # Combine all exponentials
    all_exp = np.concatenate([exp_walls, exp_pairs])
    max_exp = np.max(all_exp)
    if max_exp == 0:
        return 0.0, grad
    
    # Log-sum-exp
    shifted = all_exp / max_exp
    log_sum = np.log(np.sum(shifted)) + np.log(max_exp)
    
    # Soft min = -tau * log_sum
    obj = -tau * log_sum
    
    # Softmax weights
    weights = shifted / np.sum(shifted)
    wall_weights = weights[:len(exp_walls)]
    pair_weights = weights[len(exp_walls):]
    
    # Gradient of soft_min w.r.t. centers
    # d(soft_min)/d(c_ik) = sum over wall terms: wall_weights * d(d_wall)/d(c_ik)
    # d(d_wall_ik)/d(c_ik) = -sign(c_ik)
    sign_c = np.sign(centers)
    # Wall gradient contribution
    wall_grad = -wall_weights.reshape(n, 3) * sign_c
    grad += wall_grad
    
    # Pairwise gradient contribution
    # For pair (i,j) with weight w, contribution to grad[i] += w * (c_i - c_j) / |c_i - c_j|
    # and grad[j] += w * (c_j - c_i) / |c_i - c_j|
    # We need to map the condensed pairwise indices back
    if n > 1:
        # Build index mapping
        # pdist gives distances in order: (0,1),(0,2),...,(0,n-1),(1,2),...
        ii, jj = np.triu_indices(n, k=1)
        diffs = centers[ii] - centers[jj]
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        unit_vecs = diffs / dists_safe[:, np.newaxis]
        
        # grad[i] += sum over j>i: pair_weight * unit_vec
        # grad[j] -= sum over i<j: pair_weight * unit_vec
        grad[ii] += pair_weights[:, np.newaxis] * unit_vecs
        grad[jj] -= pair_weights[:, np.newaxis] * unit_vecs
    
    return obj, grad

def construct():
    """Construct centers of 100 equal spheres in unit cube to maximize radius."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: use a jittered grid for better starting point
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # Create grid points
    ticks = (np.arange(m) + 0.5) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to stay in cube
    centers = np.clip(centers, -0.45, 0.45)
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Annealing schedule for tau
    # Start with large tau (smooth), anneal to small tau (sharp)
    tau_start = 0.05
    tau_end = 0.001
    
    # Number of iterations - will be limited by time
    max_iter = 5000
    lr_start = 0.005
    lr_end = 0.0001
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Anneal tau
        progress = it / max_iter
        tau = tau_start * (tau_end / tau_start) ** progress
        
        # Anneal learning rate
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute objective and gradient
        obj, grad = _soft_min_obj_grad(centers, tau)
        
        # Gradient ascent
        centers = centers + lr * grad
        
        # Project back into cube (with small margin)
        centers = np.clip(centers, -0.499, 0.499)
        
        # Every 50 iterations, check actual radius
        if it % 50 == 0:
            curr_radius = _compute_radius(centers)
            if curr_radius > best_radius:
                best_radius = curr_radius
                best_centers = centers.copy()
        
        # Check time more frequently
        if time.time() > deadline - 10.0:
            break
    
    # Final check
    final_radius = _compute_radius(centers)
    if final_radius > best_radius:
        best_radius = final_radius
        best_centers = centers.copy()
    
    # One more refinement pass with smaller tau if time allows
    if time.time() < deadline - 5.0:
        tau_final = 0.0005
        for it in range(1000):
            if time.time() > deadline - 3.0:
                break
            obj, grad = _soft_min_obj_grad(best_centers, tau_final)
            lr = 0.00005
            new_centers = best_centers + lr * grad
            new_centers = np.clip(new_centers, -0.499, 0.499)
            new_radius = _compute_radius(new_centers)
            if new_radius >= best_radius:
                best_centers = new_centers
                best_radius = new_radius
    
    return best_centers
# EVOLVE-BLOCK-END
