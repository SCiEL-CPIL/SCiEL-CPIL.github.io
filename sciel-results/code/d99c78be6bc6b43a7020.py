"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the largest common radius for given centres."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    # Pairwise distances
    d = pdist(centres)
    min_pair = np.min(d)
    return min(wall_dists.min(), min_pair / 2.0)


def _energy(centres, beta=50.0):
    """
    Smooth surrogate energy: negative soft-min of pairwise distances and wall distances.
    We want to maximize the min distance, so minimize -softmin.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    min_wall_per_point = np.min(wall_dists, axis=1)
    
    # Pairwise distances
    d = pdist(centres)
    
    # Soft min of all distances (pairwise/2 and wall distances)
    all_dists = np.concatenate([d / 2.0, min_wall_per_point])
    
    # Soft min via log-sum-exp (for max, we use -logsumexp(-x/beta) * beta)
    # We want to maximize the minimum, so minimize -softmax of distances
    # softmin(x) = -logsumexp(-x/beta) * beta
    # We want to maximize softmin, so minimize -softmin = logsumexp(-x/beta) * beta
    
    # Numerically stable logsumexp
    x = -all_dists / beta
    max_x = np.max(x)
    logsumexp = max_x + np.log(np.sum(np.exp(x - max_x)))
    
    energy = -logsumexp * beta
    return energy


def _gradient_energy(centres, beta=50.0):
    """
    Gradient of the energy function.
    """
    n = len(centres)
    grad = np.zeros_like(centres)
    
    # Wall distances: for each point, min over 3 coordinates of (0.5 - |c_ik|)
    # The soft min of wall distances
    wall_dists = 0.5 - np.abs(centres)
    min_wall_per_point = np.min(wall_dists, axis=1)
    
    # Pairwise distances
    d = pdist(centres)
    
    # All distances for soft min
    all_dists = np.concatenate([d / 2.0, min_wall_per_point])
    
    # Soft min gradient
    # d/dx_i [-softmin] = d/dx_i [logsumexp(-x/beta) * beta]
    # = beta * (1/beta) * sum_j exp(-x_j/beta) / sum_j exp(-x_j/beta) * (-dx_j/dx_i)
    # = -sum_j w_j * dx_j/dx_i where w_j = exp(-x_j/beta) / sum_k exp(-x_k/beta)
    
    x = -all_dists / beta
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    w = exp_x / np.sum(exp_x)
    
    # Now compute gradient contributions
    # Pairwise distances: d_ij = |c_i - c_j|
    # dx_ij/dc_i = (c_i - c_j)/d_ij, dx_ij/dc_j = (c_j - c_i)/d_ij
    # For wall distances: d_wall_i = min_k (0.5 - |c_ik|)
    # This is more complex; for now, use numerical gradient or approximate
    
    # Let's use a simpler approach: use scipy's numerical gradient if needed
    # But for speed, let's implement a proper gradient
    
    # Actually, let's use a different approach: direct gradient computation
    
    # Gradient of -softmin = -beta * d/dx [logsumexp(-x/beta)]
    # = -beta * (1/beta) * sum_j w_j * (-dx_j/dx_i)
    # = sum_j w_j * dx_j/dx_i
    
    # Wait, let me re-derive.
    # E = -softmin(x) = logsumexp(-x/beta) * beta
    # dE/dx_i = beta * (1/beta) * sum_j exp(-x_j/beta) / sum_k exp(-x_k/beta) * (-dx_j/dx_i)
    # = -sum_j w_j * dx_j/dx_i
    
    # So grad = -sum_j w_j * dx_j/dx_i
    
    # For pairwise distances d_ij = |c_i - c_j|:
    # dx_ij/dc_i = (c_i - c_j) / d_ij
    # dx_ij/dc_j = (c_j - c_i) / d_ij
    
    # For wall distances, it's trickier. Let's handle them separately.
    
    # Pairwise part
    D = squareform(d)
    np.fill_diagonal(D, 1.0)  # avoid division by zero
    
    # w for pairwise distances
    w_pair = w[:N * (N - 1) // 2]
    
    # Build gradient from pairwise
    pair_idx = np.triu_indices(N, k=1)
    for idx in range(len(pair_idx[0])):
        i, j = pair_idx[0][idx], pair_idx[1][idx]
        diff = centres[i] - centres[j]
        dist = D[i, j]
        if dist < 1e-10:
            continue
        grad_i = w_pair[idx] * diff / dist
        grad[i] -= grad_i  # dE/dx_i = -w_j * dx_j/dx_i, and dx_ij/dc_i = diff/dist
        grad[j] += grad_i  # dx_ij/dc_j = -diff/dist, so -w_j * (-diff/dist) = w_j * diff/dist
    
    # Wall part: for each point, the soft min of 3 wall distances
    # This is complex; for simplicity, let's add a small penalty for being close to walls
    # Or we can compute the gradient numerically for the wall part
    
    # For now, add a simple wall gradient
    for i in range(N):
        for k in range(3):
            wall_d = 0.5 - np.abs(centres[i, k])
            # The contribution depends on which wall is closest
            # Approximate: push away from nearest wall
            sign = np.sign(centres[i, k])
            if sign == 0:
                sign = 1
            grad[i, k] += sign * 0.1  # small push away from walls
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initialize with a good starting configuration
    # Use a body-centered cubic-like arrangement
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Phase 1: Gradient-based optimization with smooth surrogate
    beta = 20.0
    step_size = 0.01
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Compute gradient
        grad = _gradient_energy(centres, beta)
        
        # Update
        centres = centres - step_size * grad
        
        # Project back into cube
        centres = np.clip(centres, -0.49, 0.49)
        
        # Compute current radius
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Adaptive step size
        if iteration % 50 == 0:
            step_size *= 0.95
            beta *= 1.1  # decrease beta to make soft min sharper
    
    # Phase 2: Local refinement using scipy minimize
    if time.time() < deadline - 20.0:
        def objective(x):
            c = x.reshape(-1, 3)
            return -_compute_radius(c)
        
        try:
            result = minimize(objective, best_centres.flatten(), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < -best_radius:
                candidate = result.x.reshape(-1, 3)
                candidate = np.clip(candidate, -0.49, 0.49)
                r = _compute_radius(candidate)
                if r > best_radius:
                    best_radius = r
                    best_centres = candidate
        except Exception:
            pass
    
    # Phase 3: Metropolis Monte Carlo refinement
    if time.time() < deadline - 5.0:
        temperature = 0.01
        mc_centres = best_centres.copy()
        mc_radius = best_radius
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Random perturbation
            i = np.random.randint(N)
            delta = np.random.normal(0, temperature, 3)
            new_centres = mc_centres.copy()
            new_centres[i] += delta
            new_centres[i] = np.clip(new_centres[i], -0.49, 0.49)
            
            new_radius = _compute_radius(new_centres)
            
            # Metropolis criterion
            if new_radius > mc_radius or np.random.rand() < np.exp((new_radius - mc_radius) / (temperature * 0.1)):
                mc_centres = new_centres
                mc_radius = new_radius
                if mc_radius > best_radius:
                    best_radius = mc_radius
                    best_centres = mc_centres.copy()
            
            # Anneal temperature
            temperature *= 0.999
    
    return best_centres

# EVOLVE-BLOCK-END
