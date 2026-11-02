"""Rewritten program: vectorized differentiable force-directed approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.01
    
    points = []
    k_max = int(1.0 / spacing) + 5
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    margin = spacing * 0.5
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 5
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    idx = np.argsort(radii)
    return points[idx[:n]]

def _compute_score(centers):
    """Compute the packing radius for given centers (vectorized)."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)

def _overlap_energy(x, r_target):
    """
    Compute overlap energy and gradient.
    """
    n = len(x) // 2
    centers = x.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    overlap_pairs = np.maximum(0, 2 * r_target - pair_dists)
    energy_pairs = np.sum(overlap_pairs ** 2)
    
    radii = np.linalg.norm(centers, axis=1)
    overlap_bounds = np.maximum(0, r_target + radii - 1.0)
    energy_bounds = np.sum(overlap_bounds ** 2)
    
    total_energy = energy_pairs + energy_bounds
    
    grad = np.zeros_like(x)
    
    w_pairs = 2.0 * overlap_pairs
    W_mat = np.zeros((n, n))
    W_mat[mask] = w_pairs
    W_mat = W_mat + W_mat.T
    
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad_pair = W_mat[:, :, np.newaxis] * normalized_diff
    grad_pair = np.sum(grad_pair, axis=1)
    
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    w_bounds = 2.0 * overlap_bounds
    grad_bound = w_bounds[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    grad = (grad_pair + grad_bound).flatten()
    
    return total_energy, grad

def _soft_min_objective(x, temperature):
    """
    Compute objective and gradient for soft-min packing.
    """
    n = len(x) // 2
    centers = x.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    pair_clearances = pair_dists / 2.0
    
    radii = np.linalg.norm(centers, axis=1)
    bound_clearances = 1.0 - radii
    
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    shifted = all_clearances - np.max(all_clearances)
    exp_vals = np.exp(-temperature * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    obj = - (np.max(all_clearances) - (1.0 / temperature) * np.log(sum_exp))
    
    grad = np.zeros_like(x)
    
    n_pairs = len(pair_clearances)
    w_pairs = weights[:n_pairs]
    w_bounds = weights[n_pairs:]
    
    W_mat = np.zeros((n, n))
    W_mat[mask] = w_pairs
    
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    weighted_diff = W_mat[:, :, np.newaxis] * normalized_diff
    grad_pair = -np.sum(weighted_diff, axis=1)
    
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    grad_bound = w_bounds[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    grad = (grad_pair + grad_bound).flatten()
    
    return obj, grad

def _optimize_packing(centers, max_iter=50):
    """
    Optimize packing:
    1. Anneal target radius using overlap energy minimization.
    2. Polish using soft-min objective.
    """
    x0 = centers.flatten()
    current_x = x0.copy()
    
    # Phase 1: Anneal radius
    r_start = 0.01
    r_end = 0.0531 # Just below known best
    n_steps = 10
    
    for i in range(n_steps):
        r_target = r_start + (r_end - r_start) * (i / n_steps)
        
        def scipy_obj(x):
            e, g = _overlap_energy(x, r_target)
            return e
        
        def scipy_grad(x):
            e, g = _overlap_energy(x, r_target)
            return g
            
        result = minimize(
            scipy_obj,
            current_x,
            jac=scipy_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        current_x = result.x
        
    # Phase 2: Polish with soft-min
    for temp in [50.0, 200.0]:
        def scipy_obj(x):
            o, g = _soft_min_objective(x, temp)
            return o
        
        def scipy_grad(x):
            o, g = _soft_min_objective(x, temp)
            return g
            
        result = minimize(
            scipy_obj,
            current_x,
            jac=scipy_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        new_score = _compute_score(result.x.reshape(-1, 2))
        old_score = _compute_score(current_x.reshape(-1, 2))
        
        if new_score >= old_score:
            current_x = result.x
            
    return current_x.reshape(-1, 2)

def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    """
    centers = _hexagonal_seed(N)
    optimized = _optimize_packing(centers, max_iter=50)
    
    final_score = _compute_score(optimized)
    initial_score = _compute_score(centers)
    
    if final_score < initial_score:
        optimized = centers
    
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0 - 1e-10:
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
            
    return optimized
# EVOLVE-BLOCK-END
