"""Improved program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.spatial.distance import pdist

N = 200

def construct():
    """
    Construct centers of 200 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius.
    
    Strategy:
    1. Initialize with a Fibonacci lattice (sunflower pattern).
    2. Refine using a vectorized repulsive force-based simulation.
    """
    # Step 1: Fibonacci lattice initialization
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    r = np.sqrt((i + 0.5) / N)
    theta = 2 * np.pi * (i + 0.5) / golden_ratio
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit within [-0.5, 0.5]^2
    # The max radius in the lattice is 1.0. We want to fit in a square of side 1.
    # The bounding box of the lattice is roughly [-1, 1] x [-1, 1].
    # Scale by 0.5 to fit in [-0.5, 0.5].
    scale = 0.5 / np.max(np.abs(np.column_stack([x, y])))
    x *= scale
    y *= scale
    
    centers = np.column_stack([x, y])
    
    # Ensure strictly inside
    centers = np.clip(centers, -0.4999, 0.4999)
    
    # Step 2: Repulsive force-based refinement (Vectorized)
    num_iterations = 2000
    learning_rate = 0.05
    cutoff = 0.2  # Interaction range
    
    for iteration in range(num_iterations):
        # Compute pairwise distances
        # Using pdist is efficient
        dists = pdist(centers, metric='euclidean')
        
        # We need to compute forces.
        # For each pair (i, j), if dist < cutoff, apply repulsive force.
        # This is tricky to vectorize directly into a (N, 2) force array without
        # constructing the full matrix or using advanced indexing.
        
        # Let's use the full matrix approach for clarity and speed in NumPy
        # N=200 is small enough for a 200x200 matrix.
        
        # Compute full distance matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsive forces
        # F_ij = (1 / d^2) * (c_i - c_j) / d  => (c_i - c_j) / d^3
        # Only for d < cutoff
        mask = dist_matrix < cutoff
        mask = np.where(mask, 1.0, 0.0)
        np.fill_diagonal(mask, 0.0)
        
        # Avoid division by zero
        dist_safe = np.where(dist_matrix > 1e-10, dist_matrix, 1e-10)
        
        # Force magnitude factor: 1/d^2
        # Direction: (c_i - c_j)
        # Force on i from j: (c_i - c_j) / d^3
        # Sum over j
        
        # We can compute this as:
        # F_i = sum_j (c_i - c_j) * (1/d_ij^3) * mask_ij
        # Note: (c_i - c_j) is already in diff[i, j]
        
        # diff is (N, N, 2)
        # We want to sum over axis 1 (j)
        
        # Factor = 1 / d^3
        factor = np.zeros_like(dist_matrix)
        factor[mask > 0] = 1.0 / (dist_safe[mask > 0] ** 3)
        
        # Forces from pairs: sum_j diff[i, j] * factor[i, j]
        pair_forces = np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # Distance to each boundary
        d_right = 0.5 - centers[:, 0]
        d_left = 0.5 + centers[:, 0]
        d_top = 0.5 - centers[:, 1]
        d_bottom = 0.5 + centers[:, 1]
        
        # We want to push away from boundaries if d < cutoff
        # Force is inward.
        
        # X-axis
        f_x = np.zeros(N)
        # Right wall: if d_right < cutoff, push left (negative x)
        mask_r = d_right < cutoff
        f_x[mask_r] -= 1.0 / (np.maximum(d_right[mask_r], 1e-10) ** 2)
        
        # Left wall: if d_left < cutoff, push right (positive x)
        mask_l = d_left < cutoff
        f_x[mask_l] += 1.0 / (np.maximum(d_left[mask_l], 1e-10) ** 2)
        
        # Y-axis
        f_y = np.zeros(N)
        # Top wall: if d_top < cutoff, push down (negative y)
        mask_t = d_top < cutoff
        f_y[mask_t] -= 1.0 / (np.maximum(d_top[mask_t], 1e-10) ** 2)
        
        # Bottom wall: if d_bottom < cutoff, push up (positive y)
        mask_b = d_bottom < cutoff
        f_y[mask_b] += 1.0 / (np.maximum(d_bottom[mask_b], 1e-10) ** 2)
        
        boundary_forces = np.column_stack([f_x, f_y])
        
        total_forces = pair_forces + boundary_forces
        
        # Normalize forces to prevent instability
        force_norms = np.linalg.norm(total_forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 0:
            total_forces = total_forces / max_norm * learning_rate
        else:
            total_forces = np.zeros_like(centers)
            
        centers += total_forces
        
        # Project back into square
        centers = np.clip(centers, -0.49999, 0.49999)
        
        # Anneal learning rate and cutoff
        learning_rate *= 0.999
        if iteration % 100 == 0:
            cutoff *= 0.95  # Reduce interaction range to focus on local minima
        
    return centers
