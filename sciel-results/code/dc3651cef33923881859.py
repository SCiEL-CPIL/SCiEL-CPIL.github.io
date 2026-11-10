"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centres):
    """Compute the valid packing radius for given centres in [-0.5,0.5]^3."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use broadcasting but be careful with memory for N=100 (100x100x3 = 30000, fine)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    if min_pair <= 0:
        return 0.0
    return min(min_wall, min_pair / 2.0)


def _smooth_min(values, beta):
    """Soft minimum: -1/beta * log(sum(exp(-beta * values)))."""
    # Clip to avoid overflow
    vals = np.clip(values, -1e6, 1e6)
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * vals)))


def _objective_and_grad(centres, beta):
    """
    Objective: maximize r = min(min_pair_dist/2, min_wall)
    We minimize -r using a smooth surrogate.
    Returns (objective_value, gradient).
    """
    # Wall distances: w_i = 0.5 - |c_i| for each coordinate, take min over all coords and spheres
    # Actually: wall_i = min_k (0.5 - |c_ik|)
    abs_c = np.abs(centres)
    wall_per_coord = 0.5 - abs_c  # shape (N, 3)
    # The wall constraint for sphere i is min over k of (0.5 - |c_ik|)
    # We want the minimum over all i and k
    # Smooth min over all wall distances
    wall_flat = wall_per_coord.ravel()  # shape (N*3,)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    # We want min over i<j of dist[i,j]/2
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0  # shape (N*(N-1)/2,)
    
    # Combined: we want min of all wall_flat and pair_dists
    # Use smooth min over the combined set
    all_vals = np.concatenate([wall_flat, pair_dists])
    
    # Smooth min value
    smooth_r = _smooth_min(all_vals, beta)
    
    # Gradient computation
    # d(smooth_min)/d(x_j) = -exp(-beta*x_j) / sum(exp(-beta*x_k))
    exp_neg = np.exp(-beta * np.clip(all_vals, -1e6, 1e6))
    exp_sum = np.sum(exp_neg)
    weights = exp_neg / exp_sum  # shape (len(all_vals),)
    
    grad = np.zeros_like(centres)
    
    # Wall part: first N*3 elements
    wall_weights = weights[:N * 3].reshape(N, 3)
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    # So d(smooth_min)/d(c_ik) = -wall_weights[i,k] * (-sign(c_ik)) = wall_weights[i,k] * sign(c_ik)
    # Wait: smooth_min = -1/beta * log(sum exp(-beta * v))
    # d(smooth_min)/d(v_j) = exp(-beta*v_j)/sum(exp(-beta*v_k)) = weights[j]
    # But we want to MINIMIZE -r, where r = min(...)
    # So we minimize -smooth_r... no.
    # Let's redefine: we want to maximize r. We minimize f = -r.
    # Using smooth approximation: r ≈ smooth_min(all_vals)
    # So we minimize f = -smooth_min(all_vals)
    # d f / d x = - d(smooth_min)/dx = -sum_j weights[j] * d(v_j)/dx
    
    # For wall: v = 0.5 - |c_ik|, dv/dc_ik = -sign(c_ik)
    # d f / d c_ik = - wall_weights[i,k] * (-sign(c_ik)) = wall_weights[i,k] * sign(c_ik)
    signs = np.sign(centres)
    signs[centres == 0] = 0.0
    grad += wall_weights * signs
    
    # Pair part: remaining elements
    pair_weights = weights[N * 3:]  # shape (N*(N-1)/2,)
    # For each pair (i,j) with i<j, v = dist(i,j)/2
    # d(dist)/d(c_i) = (c_i - c_j) / dist
    # d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist)
    # d(dist/2)/d(c_j) = (c_j - c_i) / (2 * dist)
    # d f / d c_i = - pair_weight * (c_i - c_j) / (2 * dist)
    # d f / d c_j = - pair_weight * (c_j - c_i) / (2 * dist)
    
    # Reconstruct indices
    idx_i, idx_j = iu
    for k in range(len(pair_weights)):
        i, j = idx_i[k], idx_j[k]
        w = pair_weights[k]
        d = dist[i, j]
        if d < 1e-10:
            continue
        # gradient contribution
        grad[i] += -w * (centres[i] - centres[j]) / (2.0 * d)
        grad[j] += -w * (centres[j] - centres[i]) / (2.0 * d)
    
    objective = -smooth_r
    return objective, grad


def _project_into_cube(centres):
    """Ensure centres are strictly inside the cube."""
    centres = np.clip(centres, -0.4999, 0.4999)
    return centres


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_centres = None
    best_radius = 0.0
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centres = grid[:N].copy()
    
    r0 = _compute_radius(init_centres)
    best_centres = init_centres.copy()
    best_radius = r0
    
    # Gradient-based optimization with adaptive beta
    beta = 50.0
    lr = 0.001
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        obj, grad = _objective_and_grad(best_centres, beta)
        
        # Gradient descent step
        new_centres = best_centres - lr * grad
        new_centres = _project_into_cube(new_centres)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius:
            best_radius = new_r
            best_centres = new_centres
            # Adaptive learning rate: increase slightly if improving
            lr = min(lr * 1.01, 0.005)
            # Increase beta to sharpen the smooth min as we converge
            if it % 100 == 0:
                beta = min(beta * 1.1, 500.0)
        else:
            # Decrease learning rate if stuck
            lr = max(lr * 0.95, 1e-5)
        
        # Early stop if very good
        if best_radius > 0.11138:
            break
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Random initialization
            random_centres = rng.uniform(-0.4, 0.4, size=(N, 3))
            random_centres = _project_into_cube(random_centres)
            r_rand = _compute_radius(random_centres)
            if r_rand > best_radius:
                best_centres = random_centres
                best_radius = r_rand
            
            # Short gradient refinement
            beta_r = 30.0
            lr_r = 0.002
            for it2 in range(1000):
                if time.time() > deadline - 5:
                    break
                obj, grad = _objective_and_grad(best_centres, beta_r)
                new_c = best_centres - lr_r * grad
                new_c = _project_into_cube(new_c)
                new_r = _compute_radius(new_c)
                if new_r > best_radius:
                    best_radius = new_r
                    best_centres = new_c
                    lr_r = min(lr_r * 1.02, 0.01)
                else:
                    lr_r = max(lr_r * 0.9, 1e-5)
    
    return best_centres
# EVOLVE-BLOCK-END
