"""Improved packing search: multi-start local optimization with vectorized gradients."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the maximum common radius for centers c (N,3)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _objective_and_grad(c):
    """
    Objective: -r (we minimize negative radius to maximize r).
    Uses a smooth approximation: soft-min of the constraints.
    Actually, let's use a direct approach: minimize the negative of the min constraint.
    For gradient, we can use the subgradient of the min.
    """
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_wall)
    
    # Gradient: subgradient of min. We'll compute gradient of -r.
    # If r is limited by a pair, gradient points away from each other.
    # If limited by wall, gradient points inward.
    # For simplicity and smoothness, let's use a soft-min.
    
    # Soft-min with temperature
    temp = 0.01
    # All constraints in terms of "slack" that we want to maximize:
    # For each pair i<j: d_ij/2
    # For each point i and coord k: 0.5 - |c_ik|
    
    # Build vector of all constraint values
    # Pair constraints
    tri_idx = np.triu_indices(N, k=1)
    pair_dists = dists[tri_idx] / 2.0
    wall_vals = wall_dists.flatten()
    
    all_vals = np.concatenate([pair_dists, wall_vals])
    
    # Soft-min
    log_sum = np.log(np.sum(np.exp(-all_vals / temp)))
    soft_min = temp * (np.log(N * (N - 1) / 2 + N * 3) - log_sum)
    # Actually, soft-min = -temp * log(sum(exp(-x/temp)))
    soft_min = -temp * np.log(np.sum(np.exp(-all_vals / temp)))
    
    # Gradient of soft-min w.r.t. c
    # d(soft_min)/d(all_vals) = exp(-all_vals/temp) / sum(exp(-all_vals/temp))
    weights = np.exp(-all_vals / temp)
    weights /= np.sum(weights)
    
    # Gradient w.r.t. pair distances
    # d(pair_dist)/d(c_i) = (c_i - c_j) / dist_ij
    grad = np.zeros_like(c)
    
    # Pair contributions
    # For each pair (i,j), weight w, gradient contribution:
    # d(soft_min)/d(c_i) += w * (c_i - c_j) / dist_ij
    # d(soft_min)/d(c_j) += w * (c_j - c_i) / dist_ij
    
    # Vectorized pair gradient
    i_idx, j_idx = tri_idx
    pair_diffs = c[i_idx] - c[j_idx]
    pair_dists_full = np.sqrt(np.sum(pair_diffs * pair_diffs, axis=1) + 1e-30)
    pair_grads = (pair_diffs / pair_dists_full[:, np.newaxis]) * weights[:len(pair_dists)][:, np.newaxis]
    
    np.add.at(grad, i_idx, pair_grads)
    np.add.at(grad, j_idx, -pair_grads)
    
    # Wall contributions
    # For each (i,k): constraint = 0.5 - |c_ik|
    # d(constraint)/d(c_ik) = -sign(c_ik)
    wall_weights = weights[len(pair_dists):]
    wall_signs = np.sign(c).flatten()
    wall_grads = -wall_weights * wall_signs
    grad += wall_grads.reshape(N, 3)
    
    # We want to maximize soft_min, so minimize -soft_min
    obj = -soft_min
    grad_obj = -grad
    
    return obj, grad_obj


def _local_optimize(c0, max_iter=2000, tol=1e-10):
    """Local optimization using L-BFGS-B."""
    res = minimize(
        _objective_and_grad,
        c0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
    )
    return res.x


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_c = None
    best_r = -np.inf
    
    # Multiple random restarts
    np.random.seed(42)
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial configuration
        # Use a jittered grid or random points
        if restart == 0:
            # Start from a grid
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            c = grid[:N].copy()
        elif restart == 1:
            # Another grid with different offset
            m = 5
            ticks = (np.arange(m) + 0.3) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            c = grid[:N].copy()
        else:
            # Random points with some structure
            # Clustered random
            n_clusters = 4
            centers = np.random.uniform(-0.4, 0.4, size=(n_clusters, 3))
            c = np.zeros((N, 3))
            for i in range(N):
                cluster = i % n_clusters
                c[i] = centers[cluster] + np.random.normal(0, 0.05, size=3)
            c = np.clip(c, -0.45, 0.45)
        
        # Local optimization
        try:
            c_opt = _local_optimize(c, max_iter=1000, tol=1e-10)
        except Exception:
            continue
        
        r = _compute_radius(c_opt)
        
        if r > best_r:
            best_r = r
            best_c = c_opt.copy()
        
        # If we've found a good radius, try a few more restarts with smaller perturbations
        if best_r > 0.11:
            # Perturb the best solution slightly
            c_perturbed = best_c + np.random.normal(0, 0.001, size=(N, 3))
            c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
            try:
                c_opt2 = _local_optimize(c_perturbed, max_iter=500, tol=1e-11)
                r2 = _compute_radius(c_opt2)
                if r2 > best_r:
                    best_r = r2
                    best_c = c_opt2.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if best_c is not None and time.time() < deadline - 10:
        try:
            c_final = _local_optimize(best_c, max_iter=3000, tol=1e-12)
            r_final = _compute_radius(c_final)
            if r_final > best_r:
                best_c = c_final
                best_r = r_final
        except Exception:
            pass
    
    # Ensure validity: clip to cube and check
    if best_c is None:
        # Fallback: simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Clip to ensure inside cube
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c

# EVOLVE-BLOCK-END
