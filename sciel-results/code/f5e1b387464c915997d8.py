"""Rewritten program: Fibonacci disk initialization + gradient ascent on smoothed min."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize using Fibonacci disk (Vogel) distribution.
    2. Apply iterative gradient ascent on the "minimum constraint" function.
       We approximate the gradient of the min function by averaging the gradients
       of the K most constraining pairs and points.
    """
    rng = np.random.default_rng(42)
    
    # --- Step 1: Initialize using Fibonacci Disk ---
    # Vogel's model for sunflower seed arrangement
    n = N
    c = 1.0  # spacing constant, adjusted for density
    # Standard Fibonacci disk: r = sqrt(i/N), theta = i * golden_angle
    # To fill the unit disk, we scale r.
    indices = np.arange(n)
    # Add a small random jitter to break symmetry and avoid getting stuck in symmetric local minima
    jitter_r = rng.random(n) * 0.01
    jitter_theta = rng.random(n) * 0.01
    
    r = np.sqrt((indices + 0.5) / n)
    # Scale to fit inside radius 1 with a small margin
    r = r * 0.95 
    r += jitter_r
    # Ensure r is not negative
    r = np.maximum(r, 0.0)
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices + jitter_theta
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # Ensure strictly inside
    radii = np.linalg.norm(centers, axis=1)
    mask = radii < 1.0
    if not np.all(mask):
        centers[mask] = centers[mask] / radii[mask, np.newaxis] * 0.99
    
    # --- Step 2: Gradient Ascent on Smoothed Min ---
    
    n_iterations = 5000
    # Adaptive learning rate
    lr = 0.05
    momentum = 0.5
    velocity = np.zeros_like(centers)
    
    # Number of "active" constraints to consider
    K = 20
    
    best_centers = centers.copy()
    best_score = -np.inf
    
    for it in range(n_iterations):
        # Compute pairwise distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Mask diagonal
        mask_diag = np.eye(n, dtype=bool)
        dist[mask_diag] = np.inf
        
        # Compute boundary distances
        radii = np.linalg.norm(centers, axis=1)
        boundary_dists = 1.0 - radii
        
        # The score is min( min_pair_dist / 2, min_boundary_dist )
        # Let's create a flat array of all "constraints"
        # Pair constraints: dist[i, j] / 2 for i < j
        # Boundary constraints: boundary_dists[i]
        
        # To save memory and time, we can just find the K smallest among all.
        # Total constraints: ~45000 pairs + 300 points = ~45300.
        
        # Get indices of K smallest pair distances
        # We need to be careful with i < j to avoid duplicates
        triu_indices = np.triu_indices(n, k=1)
        pair_dists = dist[triu_indices] / 2.0
        
        # Combine pair distances and boundary distances into a single array
        # We need to track which gradient to apply.
        # Let's just find the K smallest values from the combined set.
        
        # Create a list of (value, type, index1, index2)
        # But doing this in Python loops is slow. 
        # Instead, we can find the K smallest pair distances and K smallest boundary distances separately?
        # No, the min is global. We need the global K smallest.
        
        # Flatten pair distances
        # We can use np.argpartition on the combined array if we concatenate them.
        
        # Concatenate
        # Values: [pair_dists, boundary_dists]
        all_vals = np.concatenate([pair_dists, boundary_dists])
        
        # Find indices of K smallest
        # If K is small, argpartition is O(M)
        k_min_indices = np.argpartition(all_vals, K)[:K]
        
        # Now, for each of these K indices, determine if it's a pair or a boundary point,
        # and accumulate the gradient.
        
        grad_sum = np.zeros_like(centers)
        
        # We need to map the flat index back to the original structure.
        # The first len(pair_dists) entries are pairs.
        # The next n entries are boundaries.
        
        n_pairs = len(pair_dists)
        
        for idx in k_min_indices:
            if idx < n_pairs:
                # It's a pair
                # Recover i, j from the flat index in the upper triangle
                # This mapping is tricky. 
                # Alternative: Just compute gradients for ALL pairs and points, 
                # but only add to grad_sum if the constraint value is in the K smallest.
                # This is O(N^2) per iteration, which is 90k operations. 
                # 5000 iterations * 90k = 450M operations. 
                # NumPy vectorization is fast, but this loop is not.
                
                # Better approach:
                # 1. Compute all pair distances and boundary distances.
                # 2. Find the threshold value: the K-th smallest value.
                # 3. Create a mask of all constraints that are <= threshold.
                # 4. Compute gradients for ALL constraints, but zero out those not in the mask.
                # 5. Sum the gradients.
                
                pass # We'll implement the vectorized version below.
            
            else:
                # It's a boundary point
                pass

        # --- Vectorized Gradient Computation ---
        
        # 1. Find the threshold value
        threshold_val = np.partition(all_vals, K)[K]
        
        # 2. Mask pairs
        pair_mask = pair_dists <= threshold_val
        # 3. Mask boundaries
        boundary_mask = boundary_dists <= threshold_val
        
        # We need to accumulate gradients.
        # Pair gradients:
        # grad_i += (c_i - c_j) / (2 * d_ij)  for each j where mask is true
        # grad_j += (c_j - c_i) / (2 * d_ij)
        
        # We can use matrix operations.
        # Let D be the distance matrix.
        # Let M be the mask matrix (symmetric, 0 on diagonal).
        # We want sum_j M_ij * (c_i - c_j) / (2 * D_ij)
        
        # Construct full mask matrix
        mask_mat = np.zeros((n, n), dtype=bool)
        mask_mat[triu_indices] = pair_mask
        mask_mat = mask_mat | mask_mat.T
        
        # Avoid division by zero
        dist_safe = np.where(mask_mat & ~mask_diag, dist, 1.0)
        
        # Weight matrix: 1 / (2 * dist) where mask is true, else 0
        weights = np.where(mask_mat & ~mask_diag, 1.0 / (2.0 * dist_safe), 0.0)
        
        # Gradient contribution from pairs:
        # grad_i = sum_j weights[i, j] * (c_i - c_j)
        # = c_i * sum_j weights[i, j] - sum_j weights[i, j] * c_j
        
        sum_weights = np.sum(weights, axis=1)
        # Sum of weighted centers
        weighted_centers = weights @ centers
        
        pair_grads = centers * sum_weights[:, np.newaxis] - weighted_centers
        
        # Boundary gradients:
        # grad_i = -c_i / r_i for active boundary points
        # We only apply this if boundary_mask is true.
        
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        boundary_grads = -centers / radii_safe[:, np.newaxis]
        boundary_grads = boundary_grads * boundary_mask[:, np.newaxis]
        
        # Total gradient
        # We want to ascend the minimum, so we move in the direction of the gradient of the min.
        # The gradient of min(f) is roughly the average gradient of the active f's.
        # So we sum the gradients of the active constraints and normalize?
        # Or just sum them?
        # If we sum them, the magnitude depends on K. 
        # Let's divide by K to keep the step size consistent.
        
        total_grad = (pair_grads + boundary_grads) / K
        
        # Update with momentum
        velocity = momentum * velocity + (1.0 - momentum) * total_grad
        centers = centers + lr * velocity
        
        # Keep points inside the unit disk (hard constraint)
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 0.999
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.999
            velocity[outside] = 0.0
        
        # Compute current score
        # Reuse dist from earlier? No, centers changed.
        # But we already computed dist before the update. 
        # We need the NEW score to track best.
        # We can compute it efficiently.
        
        diff_new = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq_new = np.sum(diff_new ** 2, axis=2)
        dist_new = np.sqrt(dist_sq_new)
        np.fill_diagonal(dist_new, np.inf)
        min_pair_dist = np.min(dist_new)
        
        radii_new = np.linalg.norm(centers, axis=1)
        min_boundary_dist = 1.0 - np.max(radii_new)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Decay learning rate
        if it % 100 == 0:
            lr *= 0.95
            momentum *= 0.99
            
        if lr < 1e-6:
            break

    # Final cleanup
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
        
    return best_centers
# EVOLVE-BLOCK-END
