"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Helper to compute the exact score
    def compute_score(centers):
        # centers: (N, 3)
        # Pairwise distances
        # Use broadcasting for vectorization
        # diff[i, j, k] = centers[i, k] - centers[j, k]
        # This is N x N x 3, which is 100x100x3 = 30000 floats, manageable.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Exclude self distances (diagonal)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # Distance to each wall: 0.5 - |c|
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        # Radius is min of half pairwise dist and wall dist
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Check validity: centers must be in [-0.5, 0.5]
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            return 0.0
            
        return r

    # Soft-min objective and gradient
    def soft_min_obj_and_grad(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0) # Temporarily set to 1, we will mask later
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        
        # We need to handle the gradient carefully.
        # y_ij = d_ij / 2
        # y_ik = d_ik_wall
        
        # Weights w = exp(-y / tau)
        # Sum of weights S = sum(w)
        # Grad = - (sum w * dy/dc) / S
        
        # Calculate y values
        y_pair = dists / 2.0
        y_wall = wall_dists
        
        # Flatten y for easier handling? No, let's compute gradients directly.
        
        # Mask for pairwise (i != j)
        mask = ~np.eye(N, dtype=bool)
        
        # Weights for pairs
        # Only compute for i < j to save time? No, the formula is symmetric.
        # Let's sum over all i, j where i != j.
        
        w_pair = np.exp(-y_pair / tau)
        w_pair[~mask] = 0.0 # Zero out diagonal
        
        w_wall = np.exp(-y_wall / tau)
        
        S = np.sum(w_pair) + np.sum(w_wall)
        
        if S == 0:
            return 0.0, np.zeros_like(centers_flat)
        
        # Objective value
        obj_val = -tau * np.log(S)
        
        # Gradient
        grad = np.zeros_like(centers)
        
        # Pairwise contribution
        # grad[i] += - (1/S) * sum_j w_ij * (dy_ij/dc_i)
        # dy_ij/dc_i = (c_i - c_j) / d_ij
        
        # We can vectorize this.
        # unit_vec[i, j, k] = (c_i - c_j) / d_ij
        # Note: d_ij is dists[i,j]
        
        # Avoid div by 0 in unit vectors
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Gradient term for pairs:
        # g_pair[i] = - (1/S) * sum_j w_pair[i,j] * unit_vecs[i,j]
        # Note: w_pair is symmetric, unit_vecs[i,j] = -unit_vecs[j,i]
        
        # Sum over j for each i
        # grad_pair = - (1/S) * (w_pair @ unit_vecs) ? 
        # w_pair is (N, N). unit_vecs is (N, N, 3).
        # We want sum_j w_pair[i,j] * unit_vecs[i,j, :]
        # This is equivalent to np.sum(w_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        
        grad_pair = np.sum(w_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall contribution
        # grad_wall[i] = - (1/S) * sum_k w_wall[i,k] * (dy_ik/dc_ik)
        # dy_ik/dc_ik = sign(c_ik)
        # w_wall is (N, 3)
        
        signs = np.sign(centers)
        # If center is 0, sign is 0. Gradient is 0? 
        # Actually, d|c|/dc is sign(c). At 0 it's undefined, but we can treat as 0 or random.
        # Given the repulsive nature, centers shouldn't stay at 0.
        
        grad_wall = w_wall * signs
        
        # Combine
        grad = -(grad_pair + grad_wall) / S
        
        return obj_val, grad.flatten()

    def run_optimization(initial_centers):
        x0 = initial_centers.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Phase 1: Smooth
        tau1 = 0.1
        res1 = minimize(
            lambda x: soft_min_obj_and_grad(x, tau1)[0],
            x0,
            jac=lambda x: soft_min_obj_and_grad(x, tau1)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-4}
        )
        
        if not res1.success:
             # Try to continue from res1.x anyway if it's not terrible
             x_curr = res1.x
        else:
             x_curr = res1.x
        
        # Phase 2: Sharper
        tau2 = 0.02
        res2 = minimize(
            lambda x: soft_min_obj_and_grad(x, tau2)[0],
            x_curr,
            jac=lambda x: soft_min_obj_and_grad(x, tau2)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-5}
        )
        
        x_curr = res2.x if res2.success else x_curr
        
        # Phase 3: Very Sharp
        tau3 = 0.005
        res3 = minimize(
            lambda x: soft_min_obj_and_grad(x, tau3)[0],
            x_curr,
            jac=lambda x: soft_min_obj_and_grad(x, tau3)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-6}
        )
        
        x_final = res3.x if res3.success else x_curr
        
        return x_final.reshape(N, 3)

    # Initialization strategies
    def init_grid():
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    def init_random():
        return np.random.uniform(-0.4, 0.4, (N, 3))

    # Main Loop
    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Alternate between grid and random init
        if i % 2 == 0:
            # Perturbed grid
            centers = init_grid()
            centers += np.random.normal(0, 0.02, centers.shape)
            # Clip to bounds
            centers = np.clip(centers, -0.49, 0.49)
        else:
            # Random
            centers = init_random()
        
        optimized_centers = run_optimization(centers)
        
        # Ensure validity (clip again just in case)
        optimized_centers = np.clip(optimized_centers, -0.5, 0.5)
        
        score = compute_score(optimized_centers)
        
        if score > best_score:
            best_score = score
            best_centers = optimized_centers

    # If no valid result found (shouldn't happen), return grid
    if best_centers is None:
        return init_grid()
        
    return best_centers

# EVOLVE-BLOCK-END
