"""Rewritten program: differentiable rigid-body mechanics approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # Start with a reasonable spacing
    # For n circles in unit disk, area per circle ~ pi/n, so radius ~ sqrt(1/n)
    # Hexagonal spacing ~ 2*r
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.05  # slight overestimate to get enough points
    
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
    
    # Keep points that fit (with some margin)
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        # If not enough, reduce spacing
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
    
    # Sort by distance from center and take first n
    idx = np.argsort(radii)
    return points[idx[:n]]


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Vectorized pairwise distances
    # This is O(N^2) memory, but N=300 is small enough (90k floats)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Vectorized boundary distances
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _softmin_objective(x, t):
    """
    Compute the negative soft-min of all constraint distances.
    x: (N, 2) array of centers.
    t: temperature parameter.
    Returns: (value, gradient)
    """
    n = len(x)
    centers = x.reshape(-1, 2)
    
    # 1. Pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # 2. Boundary distances
    # dist_to_wall[i] = 1 - |centers[i]|
    # We want to maximize min(dists, dist_to_wall)
    # Let's combine all distances into a single vector D
    # D contains all pairwise distances (i<j) and all boundary distances.
    
    # To avoid storing N^2/2 + N distances explicitly in a way that's hard to gradient,
    # we compute the log-sum-exp directly.
    
    # Mask for upper triangle for pairwise
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    wall_dists = 1.0 - norms
    
    # Combine
    # We need to handle the gradient carefully.
    # Let S = sum(exp(-t * d_k))
    # val = -1/t * log(S)
    # grad_x = -1/t * (1/S) * sum(-t * exp(-t * d_k) * grad_x d_k)
    #        = sum(exp(-t * d_k) * grad_x d_k) / S
    # Note: We are minimizing -softmin, so we return gradient of -softmin?
    # scipy minimize expects gradient of the function being minimized.
    # Function to minimize: f(x) = -softmin(x)
    # grad f = - grad softmin
    # grad softmin = sum(w_k * grad d_k) where w_k = exp(-t d_k) / S
    # So grad f = - sum(w_k * grad d_k)
    
    # Calculate weights
    # To prevent underflow/overflow, shift by min dist? 
    # Actually, exp(-t * d) can underflow if t*d is large.
    # Let m = min(pair_dists, wall_dists)
    # Then exp(-t(d-m)) is safe.
    # S' = sum(exp(-t(d_k - m)))
    # w_k = exp(-t(d_k - m)) / S'
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    m = np.min(all_dists)
    
    # Compute unnormalized weights
    # For pairs:
    pair_weights = np.exp(-t * (pair_dists - m))
    # For walls:
    wall_weights = np.exp(-t * (wall_dists - m))
    
    S = np.sum(pair_weights) + np.sum(wall_weights)
    
    # Normalize
    pair_w_norm = pair_weights / S
    wall_w_norm = wall_weights / S
    
    # Objective value: -softmin = -(-1/t * log(S' * exp(-t*m))) 
    # softmin = -1/t * (log(S') - t*m) = m - 1/t * log(S')
    # f(x) = -softmin = -m + 1/t * log(S')
    val = -m + (1.0 / t) * np.log(S)
    
    # Gradient calculation
    # grad f = - grad softmin
    # grad softmin = sum_{pairs} w_pair * grad d_pair + sum_{walls} w_wall * grad d_wall
    # grad d_pair[i,j] wrt x_i = (x_i - x_j) / d_ij
    # grad d_pair[i,j] wrt x_j = (x_j - x_i) / d_ij = - (x_i - x_j) / d_ij
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    # We need to map pair_w_norm back to the (N, N) matrix or accumulate directly.
    # It's easier to create a weight matrix W where W[i,j] = pair_w_norm[idx] for i<j
    # Then grad[i] = sum_j W[i,j] * (x_i - x_j)/d_ij - sum_j W[j,i] * (x_j - x_i)/d_ij
    # Actually, let's just loop over pairs for gradient accumulation? 
    # N=300, pairs=45k. Looping 45k times in Python is slow but maybe acceptable if done once?
    # No, we want vectorization.
    
    # Let's create a full (N, N) weight matrix
    W = np.zeros((n, n))
    W[mask] = pair_w_norm
    
    # Symmetrize? No, W is upper triangle. 
    # grad[i] from pairs:
    # sum_{j != i} W[i,j] * (x_i - x_j)/d_ij  (where W[i,j] is non-zero only if i<j in our mask? No, W is only filled in upper triangle)
    # Wait, W[i,j] is non-zero for i<j.
    # For a fixed i, the terms involving x_i are:
    # 1. j < i: W[j,i] is 0 in our current W. But the pair (j,i) corresponds to W[i,j] in upper triangle? No.
    # Let's define W_sym such that W_sym[i,j] = weight for pair {i,j}.
    # W_sym = W + W.T
    
    W_sym = W + W.T
    
    # grad_pair[i] = sum_j W_sym[i,j] * (x_i - x_j) / d_ij
    # Note: d_ij is symmetric.
    
    # Vectorized computation:
    # num = sum_j W_sym[i,j] * (x_i - x_j)
    #     = x_i * sum_j W_sym[i,j] - sum_j W_sym[i,j] * x_j
    # denom = d_ij? No, the gradient of distance is (x_i - x_j) / d_ij.
    # So term is W_sym[i,j] * (x_i - x_j) / d_ij.
    
    # This is not simply linear because of the 1/d_ij.
    # We must compute: sum_j ( W_sym[i,j] / d_ij ) * (x_i - x_j)
    # Let K[i,j] = W_sym[i,j] / d_ij
    # K is symmetric.
    # grad_pair[i] = sum_j K[i,j] * (x_i - x_j)
    #              = x_i * sum_j K[i,j] - sum_j K[i,j] * x_j
    
    K = W_sym / dists
    # Handle diagonal (d=0) -> K=0
    np.fill_diagonal(K, 0.0)
    
    row_sums = np.sum(K, axis=1)
    # grad_pair = x * row_sums[:, np.newaxis] - K @ x
    grad_pair = centers * row_sums[:, np.newaxis] - K.dot(centers)
    
    # Boundary gradients
    # grad d_wall[i] = -x_i / |x_i|
    # grad_wall[i] = wall_w_norm[i] * (-x_i / |x_i|)
    
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)
    grad_wall = - (wall_w_norm / norms_safe)[:, np.newaxis] * centers
    
    # Total gradient of softmin:
    grad_softmin = grad_pair + grad_wall
    
    # Gradient of objective f(x) = -softmin:
    grad_f = -grad_softmin
    
    return val, grad_f


def _optimize_packing(centers, t=100.0, max_iter=500):
    """
    Optimize the packing using gradient descent on the negative soft-min of distances.
    """
    n = len(centers)
    x0 = centers.flatten()
    
    # We use L-BFGS-B which requires a scalar objective and gradient.
    # We wrap the softmin objective.
    
    def objective(x):
        val, _ = _softmin_objective(x, t)
        return val
    
    def gradient(x):
        _, grad = _softmin_objective(x, t)
        return grad.flatten()
    
    # Start with a higher t (sharper min) and anneal? 
    # Or just start with a reasonable t.
    # Let's try a few iterations with different t.
    
    # Stage 1: Coarse optimization
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    x_opt = result.x.reshape(-1, 2)
    
    # Stage 2: Fine optimization with higher t
    t_fine = 500.0
    def objective_fine(x):
        val, _ = _softmin_objective(x, t_fine)
        return val
        
    def gradient_fine(x):
        _, grad = _softmin_objective(x, t_fine)
        return grad.flatten()
        
    result2 = minimize(
        objective_fine,
        x_opt.flatten(),
        jac=gradient_fine,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return result2.x.reshape(-1, 2)


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses a differentiable soft-min approach to maximize the minimum clearance.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Initial score
    initial_score = _compute_score(centers)
    
    # Optimize
    optimized = _optimize_packing(centers, t=100.0, max_iter=300)
    
    # Verify the result
    final_score = _compute_score(optimized)
    
    # If the score is worse than the initial, fall back to the initial
    if final_score < initial_score:
        optimized = centers
        final_score = initial_score

    # Ensure all points are within the unit circle and non-overlapping
    # Final cleanup: clip any points that might be slightly outside
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0:
            # Scale back to the boundary
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
    
    return optimized
# EVOLVE-BLOCK-END
