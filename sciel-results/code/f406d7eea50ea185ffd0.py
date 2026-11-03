import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0, spacing_scale=1.0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    # Estimate spacing to fit N points in unit circle
    # Area ~ pi, N points. r ~ sqrt(pi/N). spacing ~ 2*r/sqrt(3) approx
    base_spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    spacing = base_spacing * spacing_scale
    
    k = int(1.0 / spacing) + 4
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                rows.append([x, y])
    
    points = np.array(rows)
    if len(points) == 0:
        # Fallback to random
        return rng.uniform(-1, 1, size=(n_points, 2))
        
    # Sort by distance to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) >= n_points:
        return points[idx[:n_points]]
    else:
        # If not enough points, add random points (should not happen with good k)
        remaining = n_points - len(points)
        rand_pts = rng.uniform(-1, 1, size=(remaining, 2))
        # Filter out points outside unit circle
        valid = np.linalg.norm(rand_pts, axis=1) <= 1.0
        rand_pts = rand_pts[valid]
        while len(rand_pts) < remaining:
            p = rng.uniform(-1, 1, size=(1, 2))
            if np.linalg.norm(p, axis=1)[0] <= 1.0:
                rand_pts = np.vstack([rand_pts, p])
        return np.vstack([points, rand_pts])

def _compute_radius(P):
    """Calculate the minimum radius for a given set of points."""
    if len(P) < 2:
        return 0.0
    # Distance between all pairs
    # P[:, np.newaxis, :] - P[np.newaxis, :, :] is N x N x 2
    # This is memory heavy for large N, but N=300 is okay (300*300*2*8 = 1.4 MB)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist_mat[iu])
    min_bound = np.min(1 - np.linalg.norm(P, axis=1))
    return min(min_dist / 2.0, min_bound)

def _soft_min_logsumexp(x, beta):
    """
    Soft minimum using log-sum-exp.
    min(x) ~ -1/beta * log(sum(exp(-beta * x)))
    """
    # Shift for numerical stability
    x_shifted = x - np.min(x)
    return -1.0/beta * np.log(np.sum(np.exp(-beta * x_shifted))) - np.min(x)

def _objective_and_grad(flat_P, beta):
    """
    Objective function to minimize: -soft_min(distances/2, boundary_distances)
    Gradient: d/dP of the objective.
    """
    P = flat_P.reshape(N, 2)
    
    # Calculate pairwise distances
    # To save memory and time, we can compute gradients analytically for the soft min
    # Let D be the matrix of distances.
    # SoftMin = -1/beta * log(sum(exp(-beta * D)))
    # Grad SoftMin w.r.t D_k = exp(-beta * D_k) / sum(exp(-beta * D))
    
    # We need distances between all pairs (i, j) with i < j
    # And boundary distances (1 - |P_i|)
    
    # Vectorized distance calculation
    # diff[i, j, :] = P[i] - P[j]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dist_mat[mask]
    pair_diffs = diff[mask] # Shape: (M, 2) where M = N*(N-1)/2
    
    # Boundary distances
    norms = np.linalg.norm(P, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all distances into one vector
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Calculate soft min and weights
    # We want to maximize the minimum distance.
    # So we minimize -soft_min(all_dists)
    
    # For numerical stability in logsumexp
    min_d = np.min(all_dists)
    all_dists_shifted = all_dists - min_d
    
    exp_vals = np.exp(-beta * all_dists_shifted)
    sum_exp = np.sum(exp_vals)
    
    # Weights for each distance term
    weights = exp_vals / sum_exp
    
    # Objective value: -soft_min
    # soft_min = min_d - 1/beta * log(sum_exp)
    # -soft_min = -min_d + 1/beta * log(sum_exp)
    # Note: We are minimizing this.
    obj = -min_d + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # d(soft_min)/d(x_k) = -weights_k
    # d(-soft_min)/d(x_k) = weights_k
    
    # We need d(obj)/d(P)
    # obj depends on P through all_dists.
    # d(obj)/d(P) = sum_k weights_k * d(x_k)/d(P)
    
    grad_P = np.zeros_like(P)
    
    # 1. Pairwise distances contribution
    # x_k = |P_i - P_j|
    # d(x_k)/d(P_i) = (P_i - P_j) / |P_i - P_j|
    # d(x_k)/d(P_j) = (P_j - P_i) / |P_i - P_j| = - (P_i - P_j) / |P_i - P_j|
    
    # We have pair_dists and pair_diffs (P_i - P_j)
    # We need to map pair indices back to (i, j)
    # This is tricky to do vectorized without explicit loops or complex indexing.
    # Given N=300, M=44850. A loop over M is slow in Python.
    # Let's try to vectorize the gradient accumulation.
    
    # We can use np.add.at to accumulate gradients
    # We need the indices i and j for each pair in the flattened array.
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # unit_vecs = pair_diffs / pair_dists[:, np.newaxis]
    # Handle division by zero (if points coincide)
    safe_dists = np.where(pair_dists > 1e-8, pair_dists, 1.0)
    unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
    
    # Gradient contribution for each pair k:
    # w_k * unit_vec_k for P_i
    # -w_k * unit_vec_k for P_j
    
    # Weights for pairs are the first M elements of weights
    w_pairs = weights[:len(pair_dists)]
    
    # Accumulate to P[i]
    # np.add.at(grad_P, i_idx, w_pairs[:, np.newaxis] * unit_vecs)
    # np.add.at(grad_P, j_idx, -w_pairs[:, np.newaxis] * unit_vecs)
    
    # This might be slow. Let's check if we can do it faster.
    # For N=300, this is acceptable in a tight loop if optimized, but scipy calls this many times.
    # Alternative: Use a sparse matrix or explicit loops if necessary.
    # Let's stick to np.add.at for now.
    
    np.add.at(grad_P, i_idx, w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad_P, j_idx, -w_pairs[:, np.newaxis] * unit_vecs)
    
    # 2. Boundary distances contribution
    # x_k = 1 - |P_i|
    # d(x_k)/d(P_i) = - P_i / |P_i|
    
    w_bounds = weights[len(pair_dists):]
    
    # unit_norms = P / norms[:, np.newaxis]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    unit_norms = P / safe_norms[:, np.newaxis]
    
    # grad_P += w_bounds[:, np.newaxis] * (-unit_norms)
    grad_P -= w_bounds[:, np.newaxis] * unit_norms
    
    return obj, grad_P.flatten()

def _force_relax(P, iterations=50, repulsion=1.0, attraction=0.1, max_step=0.01):
    """
    Simple force-directed relaxation.
    """
    P = P.copy()
    for _ in range(iterations):
        forces = np.zeros_like(P)
        
        # Pairwise repulsion
        # We only need to compute forces for close pairs to save time?
        # For N=300, full matrix is fine.
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist_mat[dist_mat < 1e-6] = 1e-6
        
        # Force = repulsion / dist^2 * unit_vec
        # We want to push apart if dist < 2*r (approx).
        # Let's use a smooth kernel.
        # F ~ 1/d^2
        inv_dist_sq = 1.0 / (dist_mat ** 2)
        
        # Zero out diagonal
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Multiply by direction
        # forces[i] += sum_j repulsion * inv_dist_sq[i,j] * (P[i] - P[j]) / dist[i,j]
        # = sum_j repulsion * inv_dist_sq[i,j] * diff[i,j] / dist[i,j]
        # = sum_j repulsion * inv_dist_sq[i,j] * diff[i,j] * inv_dist[i,j]
        
        # Let's compute unit vectors
        # unit_diff = diff / dist_mat[:, :, np.newaxis]
        # This is heavy.
        
        # Alternative:
        # force_vec = diff * (inv_dist_sq / dist_mat)[:, :, np.newaxis]
        # factor = inv_dist_sq / dist_mat
        # factor[dist_mat < 1e-6] = 0
        factor = inv_dist_sq / dist_mat
        np.fill_diagonal(factor, 0.0)
        
        repulsive_forces = diff * factor[:, :, np.newaxis]
        forces += np.sum(repulsive_forces, axis=1) * repulsion
        
        # Attraction to center
        forces -= P * attraction
        
        # Limit step size
        step_norms = np.linalg.norm(forces, axis=1)
        max_step_vec = max_step * np.ones_like(step_norms)
        # Normalize forces if step > max_step
        scale = np.where(step_norms > max_step_vec, max_step_vec / step_norms, 1.0)
        forces *= scale[:, np.newaxis]
        
        P += forces
        
        # Project back to unit circle
        norms = np.linalg.norm(P, axis=1)
        out_of_bounds = norms > 1.0
        if np.any(out_of_bounds):
            P[out_of_bounds] /= norms[out_of_bounds, np.newaxis]
            
    return P

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_P = None
    best_radius = 0.0
    
    # Parameters for soft min
    # Beta controls the "sharpness" of the min.
    # Start with lower beta (smoother) and increase?
    # Or just use a fixed moderate beta.
    beta = 50.0
    
    # Multi-restart
    num_restarts = 10
    rng = np.random.default_rng(42)
    
    for k in range(num_restarts):
        if time.time() - start_time > time_limit - 20.0:
            break
            
        # Generate initial points
        # Vary the spacing scale to explore different densities/structures
        spacing_scale = 0.9 + 0.1 * k / (num_restarts - 1)
        P_init = _hex_init(N, seed=k, spacing_scale=spacing_scale)
        
        # Add small random noise to break symmetry
        P_init += rng.normal(0, 0.01, size=(N, 2))
        
        # Ensure inside unit circle
        norms = np.linalg.norm(P_init, axis=1)
        out = norms > 1.0
        if np.any(out):
            P_init[out] /= norms[out, np.newaxis]
            
        # Force Relaxation
        P_relaxed = _force_relax(P_init, iterations=50)
        
        # Optimize using L-BFGS-B
        # We minimize -soft_min(radius)
        # Objective returns obj, grad
        # We need to wrap it to handle beta
        
        def obj_wrapper(flat_P):
            obj, grad = _objective_and_grad(flat_P, beta)
            return obj, grad
            
        try:
            res = minimize(
                obj_wrapper, 
                P_relaxed.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            P_opt = res.x.reshape(N, 2)
        except Exception as e:
            P_opt = P_relaxed
            
        # Calculate actual radius
        r = _compute_radius(P_opt)
        
        if r > best_radius:
            best_radius = r
            best_P = P_opt.copy()
            
    return best_P

# EVOLVE-BLOCK-END
