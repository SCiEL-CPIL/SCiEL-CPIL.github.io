"""Rewritten program: Fibonacci disk initialization + gradient ascent on smoothed min."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize using Fibonacci disk (Vogel) distribution with better scaling.
    2. Apply iterative gradient ascent on the "minimum constraint" function
       using K most constraining pairs and points.
    3. Use adaptive learning rate with momentum for faster convergence.
    """
    rng = np.random.default_rng(42)
    n = N
    
    # --- Step 1: Initialize using Fibonacci Disk ---
    # Better initialization: use golden angle with proper radial distribution
    indices = np.arange(n)
    
    # Use a slightly compressed radial distribution to allow room for optimization
    # r = sqrt((i + 0.5) / n) * scale_factor
    # We want the outermost point to be at about 0.98 to leave room
    r = np.sqrt((indices + 0.5) / n) * 0.98
    
    # Add small random perturbation to break symmetry
    r += rng.random(n) * 0.005
    r = np.clip(r, 0.0, 0.99)
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices + rng.random(n) * 0.01
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # Ensure strictly inside
    radii = np.linalg.norm(centers, axis=1)
    mask = radii >= 1.0
    if np.any(mask):
        centers[mask] = centers[mask] / radii[mask, np.newaxis] * 0.99
    
    # --- Step 2: Gradient Ascent on Smoothed Min ---
    
    n_iterations = 8000
    lr = 0.03
    momentum = 0.7
    velocity = np.zeros_like(centers)
    
    # Number of "active" constraints to consider (start small, increase later)
    K = 15
    
    best_centers = centers.copy()
    best_score = -np.inf
    no_improve_count = 0
    patience = 500
    
    for it in range(n_iterations):
        # Adaptive K: increase as we progress
        if it < 2000:
            K = 10
        elif it < 4000:
            K = 20
        elif it < 6000:
            K = 30
        else:
            K = 40
        
        # Compute pairwise distances using cdist for efficiency
        dist = cdist(centers, centers, metric='euclidean')
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        # Compute boundary distances
        radii = np.linalg.norm(centers, axis=1)
        boundary_dists = 1.0 - radii
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n, k=1)
        pair_dists = dist[triu_indices] / 2.0
        
        # Combine pair distances and boundary distances
        all_vals = np.concatenate([pair_dists, boundary_dists])
        
        # Find the K-th smallest value as threshold
        threshold_val = np.partition(all_vals, K)[K]
        
        # Create masks for active constraints
        pair_mask = pair_dists <= threshold_val
        boundary_mask = boundary_dists <= threshold_val
        
        # Construct full mask matrix for pairs
        mask_mat = np.zeros((n, n), dtype=bool)
        mask_mat[triu_indices] = pair_mask
        mask_mat = mask_mat | mask_mat.T
        
        # Avoid division by zero
        dist_safe = np.where(mask_mat, dist, 1.0)
        dist_safe = np.where(dist_safe < 1e-10, 1e-10, dist_safe)
        
        # Weight matrix: 1 / (2 * dist) where mask is true, else 0
        weights = np.where(mask_mat, 1.0 / (2.0 * dist_safe), 0.0)
        np.fill_diagonal(weights, 0.0)
        
        # Gradient contribution from pairs:
        # grad_i = sum_j weights[i, j] * (c_i - c_j)
        sum_weights = np.sum(weights, axis=1)
        weighted_centers = weights @ centers
        pair_grads = centers * sum_weights[:, np.newaxis] - weighted_centers
        
        # Boundary gradients:
        # For points near boundary, gradient pushes them inward
        radii_safe = np.where(radii > 1e-8, radii, 1.0)
        boundary_grads = -centers / radii_safe[:, np.newaxis]
        boundary_grads = boundary_grads * boundary_mask[:, np.newaxis]
        
        # Total gradient: average of active constraint gradients
        total_grad = (pair_grads + boundary_grads) / max(K, 1)
        
        # Update with momentum
        velocity = momentum * velocity + (1.0 - momentum) * total_grad
        centers = centers + lr * velocity
        
        # Keep points inside the unit disk (hard constraint)
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 0.9995
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.9995
            velocity[outside] = 0.0
        
        # Compute current score
        dist_new = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dist_new, np.inf)
        min_pair_dist = np.min(dist_new)
        
        radii_new = np.linalg.norm(centers, axis=1)
        min_boundary_dist = 1.0 - np.max(radii_new)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Decay learning rate
        if it % 50 == 0:
            lr *= 0.97
            momentum = 0.7 + 0.1 * (1.0 - it / n_iterations)
        
        # Early stopping if no improvement for long time
        if no_improve_count > patience and it > 1000:
            break
            
        if lr < 1e-7:
            break

    # Final cleanup
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
        
    return best_centers
# EVOLVE-BLOCK-END
