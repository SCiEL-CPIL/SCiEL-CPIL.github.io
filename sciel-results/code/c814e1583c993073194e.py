"""Rewritten program: vectorized differentiable force-directed approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # Start with a reasonable spacing
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.05
    
    # Generate hexagonal lattice points
    points = []
    k_max = int(1.0 / spacing) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points that fit
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 3
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
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Mask out self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Boundary distance
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _objective_and_grad(x, temperature):
    """
    Compute objective and gradient for the differentiable packing problem.
    Objective: -soft_min(min_clearance)
    We want to maximize min_clearance, so we minimize -min_clearance.
    """
    n = len(x) // 2
    centers = x.reshape(n, 2)
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    # We only care about i < j. 
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract upper triangle distances
    pair_dists = dists[mask]
    # Clearances from pairs: d_ij / 2
    pair_clearances = pair_dists / 2.0
    
    # 2. Boundary clearances
    radii = np.linalg.norm(centers, axis=1)
    bound_clearances = 1.0 - radii
    
    # Combine all clearances
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Objective: -soft_min(all_clearances)
    # Gradient of soft_min(z) wrt z_k is exp(-T*z_k) / sum(exp(-T*z_j))
    
    shifted = all_clearances - np.max(all_clearances)
    exp_vals = np.exp(-temperature * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # The objective is -soft_min.
    # d(-soft_min)/d z_k = -weights[k]
    
    obj = - (np.max(all_clearances) - (1.0 / temperature) * np.log(sum_exp))
    
    grad = np.zeros_like(x)
    
    # 3. Gradient for Pairwise Clearances
    # z_k corresponds to d_ij / 2
    # d z_k / d centers[i] = (centers[i] - centers[j]) / (2 * d_ij)
    # d z_k / d centers[j] = (centers[j] - centers[i]) / (2 * d_ij)
    
    n_pairs = len(pair_clearances)
    w_pairs = weights[:n_pairs]
    w_bounds = weights[n_pairs:]
    
    # Reconstruct the full weight matrix for pairs
    W_mat = np.zeros((n, n))
    W_mat[mask] = w_pairs
    
    # Gradient contribution from pairs
    # d(-soft_min)/d centers[i] = sum_j ( -w_ij * d(d_ij/2)/d centers[i] )
    # = sum_j ( -w_ij * (centers[i] - centers[j]) / d_ij )
    
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    weighted_diff = W_mat[:, :, np.newaxis] * normalized_diff
    grad_pair = -np.sum(weighted_diff, axis=1)
    
    # 4. Gradient for Boundary Clearances
    # z_k = 1 - |c_k|
    # d z_k / d c_k = - c_k / |c_k|
    # d(-soft_min)/d c_k = - w_k * (- c_k / |c_k|) = w_k * c_k / |c_k|
    
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    grad_bound = w_bounds[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    # Total Gradient
    grad = grad_pair + grad_bound
    
    # Flatten grad
    grad = grad.flatten()
    
    return obj, grad


def _optimize_packing(centers, max_iter=500):
    """
    Optimize the packing using L-BFGS-B with annealing temperature.
    """
    x0 = centers.flatten()
    
    current_x = x0.copy()
    
    # Annealing schedule
    for temp in [10.0, 50.0, 200.0, 1000.0]:
        def scipy_obj(x):
            obj, grad = _objective_and_grad(x, temp)
            return obj
        
        def scipy_grad(x):
            obj, grad = _objective_and_grad(x, temp)
            return grad
            
        result = minimize(
            scipy_obj,
            current_x,
            jac=scipy_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter // 4, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        # Only accept if score improved or stayed similar
        new_score = _compute_score(result.x.reshape(-1, 2))
        old_score = _compute_score(current_x.reshape(-1, 2))
        
        if new_score >= old_score - 1e-8:
            current_x = result.x
        else:
            pass
            
    return current_x.reshape(-1, 2)


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Optimize
    optimized = _optimize_packing(centers, max_iter=1000)
    
    # Verify and cleanup
    final_score = _compute_score(optimized)
    initial_score = _compute_score(centers)
    
    if final_score < initial_score:
        optimized = centers
    
    # Ensure points are strictly inside
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0 - 1e-10:
            # Push back slightly
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
            
    return optimized
# EVOLVE-BLOCK-END
