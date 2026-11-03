"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    # Compute pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to walls
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_obj_and_grad(centers, beta):
    """Soft-min surrogate: minimize -softmin of all constraints.
    
    Constraints: d_ij/2 for all pairs, and wall distances.
    softmin(x) = -(1/beta) * log(sum(exp(-beta*x)))
    We minimize -softmin = (1/beta) * log(sum(exp(-beta*x)))
    """
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-16)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # All constraint values
    # Pair constraints: d_ij / 2
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    
    # Wall constraints: 0.5 - |c_ik| for each sphere and each coordinate
    wall_vals = wall_dists.flatten()
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft-min: -(1/beta) * log(sum(exp(-beta * vals)))
    # We want to maximize soft-min, so minimize -soft-min
    # -soft-min = (1/beta) * log(sum(exp(-beta * vals)))
    
    # For numerical stability, use log-sum-exp trick
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_terms))
    
    obj = (1.0 / beta) * (log_sum - beta * max_val)
    # Wait, let me redo this. 
    # softmin = -(1/beta) * log(sum(exp(-beta * x)))
    # We want to minimize -softmin = (1/beta) * log(sum(exp(-beta * x)))
    # For stability: log(sum(exp(-beta * x))) = -beta * max_x + log(sum(exp(-beta * (x - max_x))))
    # So obj = (1/beta) * (-beta * max_x + log(sum(exp(-beta * (x - max_x)))))
    #       = -max_x + (1/beta) * log(sum(exp(-beta * (x - max_x))))
    
    obj = -max_val + (1.0 / beta) * log_sum
    
    # Gradient
    # d/dx_j [-(1/beta) * log(sum_i exp(-beta * x_i))]
    # = -(1/beta) * [1/sum_i exp(-beta * x_i)] * sum_i (-beta * exp(-beta * x_i)) * d x_i / d x_j
    # For our case, each constraint x_i depends on centers.
    
    # Let's compute gradient differently.
    # f = (1/beta) * log(sum_k exp(-beta * c_k)) where c_k are constraints
    # df/dc_k = (1/beta) * [exp(-beta * c_k) / sum_j exp(-beta * c_j)] * (-beta)
    #          = -exp(-beta * c_k) / sum_j exp(-beta * c_j)
    #          = -softmax(-beta * c)_k
    
    # But we need d f / d centers, so use chain rule.
    
    # First, compute weights
    weights = exp_terms / np.sum(exp_terms)  # softmax(-beta * (all_vals - max_val))
    # Note: softmax(-beta * c) = exp(-beta * c) / sum(exp(-beta * c))
    # Our weights are exp(-beta * (c - max)) / sum(exp(-beta * (c - max)))
    # = exp(-beta * c + beta * max) / sum(exp(-beta * c + beta * max))
    # = exp(-beta * c) / sum(exp(-beta * c))  (the beta*max cancels)
    # So weights_k = exp(-beta * c_k) / sum_j exp(-beta * c_j)
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    # For pair (i,j), constraint c = d_ij / 2
    # d c / d c_i = -(c_i - c_j) / d_ij
    # d c / d c_j = (c_i - c_j) / d_ij
    # d f / d c_i = weights_pair * d c / d c_i
    
    pair_weights = weights[:len(pair_vals)]
    
    for idx in range(len(pair_vals)):
        i, j = iu[0][idx], iu[1][idx]
        w = pair_weights[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        # d(c_ij/2)/d(c_i) = -(c_i - c_j) / (2 * d) * 2 = -(c_i - c_j) / d
        # Wait: c = d_ij / 2, d_ij = ||c_i - c_j||
        # d c_ij / d c_i = (c_i - c_j) / d_ij
        # So d(c/2)/d c_i = (c_i - c_j) / (2 * d_ij)
        
        grad[i] += w * (centers[i] - centers[j]) / (2.0 * d)
        grad[j] += w * (centers[j] - centers[i]) / (2.0 * d)
    
    # Wall contributions
    # For sphere i, coordinate k: constraint c = 0.5 - |c_ik|
    # d c / d c_ik = -sign(c_ik)
    
    wall_weights = weights[len(pair_vals):]
    wall_grad = np.zeros_like(centers)
    
    signs = np.sign(centers)
    signs[centers == 0] = 1.0
    
    # wall_vals is flattened wall_dists, shape (n*3)
    # wall_weights has same shape
    # For each element, d f / d c_ik = wall_weights * (-sign(c_ik))
    wall_grad = -np.outer(wall_weights.reshape(n, 3), signs)
    
    grad += wall_grad
    
    return obj, grad

def _polish_obj_and_grad(centers):
    """Polish: maximize min distance using exact local search.
    
    Use a high-precision approach: find the bottleneck constraints and optimize.
    """
    n = len(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-16)
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = 0.5 - np.abs(centers)
    
    # Find minimum constraint
    iu = np.triu_indices(n, k=1)
    min_pair_idx = np.argmin(dists[iu])
    i_min, j_min = iu[0][min_pair_idx], iu[1][min_pair_idx]
    min_pair_val = dists[i_min, j_min] / 2.0
    
    min_wall_idx = np.argmin(wall_dists.flatten())
    min_wall_val = wall_dists.flatten()[min_wall_idx]
    
    current_min = min(min_pair_val, min_wall_val)
    
    # Objective: -current_min (we want to maximize it)
    obj = -current_min
    
    # Gradient
    grad = np.zeros_like(centers)
    
    if min_pair_val <= min_wall_val:
        # Bottleneck is pair (i_min, j_min)
        d = dists[i_min, j_min]
        if d > 1e-10:
            direction = (centers[i_min] - centers[j_min]) / d
            grad[i_min] += direction / 2.0
            grad[j_min] -= direction / 2.0
    else:
        # Bottleneck is wall
        sphere_idx = min_wall_idx // 3
        coord_idx = min_wall_idx % 3
        sign = np.sign(centers[sphere_idx, coord_idx])
        if sign == 0:
            sign = 1.0
        grad[sphere_idx, coord_idx] -= sign
    
    return obj, grad

def _optimize_with_bfgs(centers_init, obj_func, max_iter=200, tol=1e-12):
    """Optimize using BFGS."""
    try:
        result = minimize(
            obj_func,
            centers_init.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'disp': False}
        )
        return result.x.reshape(-1, 3), result.success, result.fun
    except Exception:
        return centers_init, False, np.inf

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    # Initial best from grid
    grid_init = _grid_init(N)
    r_grid = _compute_radius(grid_init)
    best_radius = r_grid
    best_centers = grid_init.copy()
    
    # Multi-scale soft-min optimization
    # Start with low beta (smooth) to find good global structure
    # Then increase beta to sharpen toward the true min
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Try multiple random starts
    for seed in range(8):
        if time.time() > deadline - 25.0:
            break
            
        init = _random_init(N, seed=seed)
        current_centers = init
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            def obj_and_grad(x, b=beta):
                c = x.reshape(-1, 3)
                o, g = _soft_min_obj_and_grad(c, b)
                return o, g.flatten()
            
            try:
                optimized, success, obj = _optimize_with_bfgs(
                    current_centers, obj_and_grad, max_iter=150, tol=1e-10
                )
                current_centers = optimized
            except Exception:
                continue
            
            r_opt = _compute_radius(current_centers)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = current_centers.copy()
        
        # Use best so far as next start for better convergence
        if best_centers is not None:
            init = best_centers.copy()
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5.0:
        def polish_obj_and_grad(x):
            c = x.reshape(-1, 3)
            o, g = _polish_obj_and_grad(c)
            return o, g.flatten()
        
        try:
            optimized, success, obj = _optimize_with_bfgs(
                best_centers, polish_obj_and_grad, max_iter=300, tol=1e-14
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
