"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    # Walls are at -0.5 and 0.5
    wall_dists = np.min(np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ]), axis=1)
    
    if np.any(wall_dists < -1e-10):
        return 0.0
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _softmin_grad(vals, beta):
    """
    Compute soft-min of vals and its gradient.
    softmin(vals, beta) = -1/beta * log(sum(exp(-beta * vals)))
    grad_i = exp(-beta * vals_i) / sum(exp(-beta * vals))
    """
    # Shift for numerical stability
    vals_shifted = vals - np.min(vals)
    exp_vals = np.exp(-beta * vals_shifted)
    sum_exp = np.sum(exp_vals)
    # Soft-min value
    soft_min_val = -1.0 / beta * np.log(sum_exp) + np.min(vals)
    # Gradient w.r.t vals
    grad = exp_vals / sum_exp
    return soft_min_val, grad

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Initial Grid Placement
    # 4x5x5 = 100
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    
    # Optimization Function
    def objective_and_grad(x_flat, beta):
        centers = x_flat.reshape(N, 3)
        
        # Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Get unique pairs i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Wall distances (raw, not min'd yet for gradient smoothness)
        W = np.zeros((N, 6))
        W[:, 0] = centers[:, 0] + 0.5
        W[:, 1] = -centers[:, 0] + 0.5
        W[:, 2] = centers[:, 1] + 0.5
        W[:, 3] = -centers[:, 1] + 0.5
        W[:, 4] = centers[:, 2] + 0.5
        W[:, 5] = -centers[:, 2] + 0.5
        
        V_pairs = pair_dists
        V_walls_flat = W.flatten()
        V_all = np.concatenate([V_pairs, V_walls_flat])
        
        val, grad_all = _softmin_grad(V_all, beta)
        obj = -val
        
        grad_flat = np.zeros_like(x_flat)
        
        # 1. Pair contributions
        grad_pair = grad_all[:len(V_pairs)]
        diffs = centers[i_idx] - centers[j_idx]
        norms = np.linalg.norm(diffs, axis=1)
        norms[norms == 0] = 1e-8
        unit_vecs = diffs / norms[:, np.newaxis]
        
        np.add.at(grad_flat, i_idx, grad_pair[:, None] * 0.5 * unit_vecs)
        np.add.at(grad_flat, j_idx, -grad_pair[:, None] * 0.5 * unit_vecs)
        
        # 2. Wall contributions
        grad_wall_flat = grad_all[len(V_pairs):]
        grad_wall = grad_wall_flat.reshape(N, 6)
        
        grad_flat[:, 0] += grad_wall[:, 0] - grad_wall[:, 1]
        grad_flat[:, 1] += grad_wall[:, 2] - grad_wall[:, 3]
        grad_flat[:, 2] += grad_wall[:, 4] - grad_wall[:, 5]
        
        return obj, grad_flat

    best_centers = grid.copy()
    best_radius = _compute_radius(best_centers)
    
    # Multi-stage optimization with symmetry breaking
    # We will try a few different random perturbations and betas
    rng = np.random.default_rng(42)
    
    # List of (beta, max_iter) pairs to try in sequence for each restart
    # Start smooth, end sharp
    beta_schedule = [10, 50, 100, 500, 1000, 2000]
    
    num_restarts = 5
    time_per_restart = 20.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb the grid to break symmetry
        # The perturbation size should be small enough to stay in the same basin
        # but large enough to break the grid symmetry
        perturbation = rng.normal(0, 0.01, size=(N, 3))
        current_centers = best_centers + perturbation
        # Ensure they stay within bounds
        current_centers = np.clip(current_centers, -0.49, 0.49)
        
        current_radius = _compute_radius(current_centers)
        
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            
            x0 = current_centers.flatten()
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    x0,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if res.success or res.fun < 0:
                    new_centers = res.x.reshape(N, 3)
                    new_radius = _compute_radius(new_centers)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                        current_centers = new_centers.copy()
            except Exception:
                pass

    return best_centers
# EVOLVE-BLOCK-END
