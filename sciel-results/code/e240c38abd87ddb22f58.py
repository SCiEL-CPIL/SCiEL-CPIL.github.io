"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize with a Fibonacci spiral distribution in the disk.
    2. Apply iterative relaxation using repulsive forces (inverse square law)
       and boundary constraints.
    3. Use an adaptive step size to ensure convergence to a local optimum.
    4. Return the best configuration found.
    """
    rng = np.random.default_rng(42)
    
    # --- Step 1: Initialize with Fibonacci spiral in disk ---
    # Use golden angle for uniform angular distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(N)
    # Radial positions: uniform area density -> r = sqrt((i + 0.5) / N)
    r = np.sqrt((indices + 0.5) / N)
    theta = golden_angle * indices
    
    # Add a small random perturbation to break symmetry and avoid local minima
    # The Fibonacci spiral is deterministic, so adding noise helps escape symmetry traps
    noise_r = rng.random(N) * 0.01
    noise_theta = rng.random(N) * 0.01
    r = r + noise_r
    theta = theta + noise_theta
    
    # Clamp r to be within [0, 1)
    r = np.clip(r, 0.0, 0.99)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # --- Step 2: Repulsion-based relaxation ---
    # We minimize the potential energy E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
    # Gradient descent on this potential.
    
    n_iterations = 30000
    initial_step = 0.02
    decay = 0.9999
    min_step = 1e-6
    
    step_size = initial_step
    best_centers = centers.copy()
    best_score = -np.inf
    
    for it in range(n_iterations):
        # Compute pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute repulsive forces
        # F_ij = -dV/dx = 2 * (c_i - c_j) / d_ij^4 * 0.5? 
        # V = 1/d^2 = d^-2. dV/dd = -2 d^-3. 
        # Force = -grad V. 
        # Let's just use F = diff / dist^4 which is proportional to gradient of 1/d^2.
        # Actually, let's use a simpler force: F = diff / dist^3 (inverse square magnitude, unit vector direction)
        # This corresponds to V = 1/d.
        # V = 1/d is softer and allows better global optimization than 1/d^2.
        
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, 1.0)
        
        # Force magnitude: 1 / dist^2
        # Direction: diff / dist
        # Total Force: diff / dist^3
        force_mag = 1.0 / (dist ** 3)
        np.fill_diagonal(force_mag, 0.0)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force
        # Push points inward if they are close to the boundary
        radii = np.linalg.norm(centers, axis=1)
        # We want to maximize distance to boundary, so minimize 1/(1-r)^2
        # Force is inward, proportional to 1/(1-r)^3
        # To avoid singularity, clamp radius
        radii_clamped = np.clip(radii, 0.0, 0.999)
        boundary_force_mag = 1.0 / ((1.0 - radii_clamped) ** 3)
        
        # Direction is inward: -centers / |centers|
        # If center is 0, direction is 0
        safe_radii = np.where(radii > 1e-8, radii, 1.0)
        boundary_forces = -centers / safe_radii[:, np.newaxis] * boundary_force_mag[:, np.newaxis]
        
        total_forces = forces + boundary_forces
        
        # Update positions
        centers = centers + step_size * total_forces
        
        # Project back to unit disk if outside
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 1.0
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.9999
        
        # Compute current score
        dist_new = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair_dist = np.min(dist_new)
        min_boundary_dist = 1.0 - np.max(np.linalg.norm(centers, axis=1))
        current_score = min(min_pair_dist, min_boundary_dist)
        
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Decay step size
        step_size *= decay
        if step_size < min_step:
            break
    
    # Final cleanup
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.9999
        
    return best_centers
# EVOLVE-BLOCK-END
