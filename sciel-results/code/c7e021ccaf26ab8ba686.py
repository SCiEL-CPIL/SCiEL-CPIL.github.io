"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize with a Fibonacci-like spiral distribution in the disk for uniformity.
    2. Apply iterative relaxation:
       - Compute forces from pairwise repulsion (soft penalty for overlaps)
       - Add boundary constraint force
       - Update positions with adaptive step size
    3. Return the final positions.
    """
    rng = np.random.default_rng(42)
    
    # --- Step 1: Initialize with Fibonacci spiral in disk ---
    # Use golden angle for uniform angular distribution, and sqrt distribution for radial
    # to ensure uniform area density
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(N)
    # Radial positions: uniform in area -> r = sqrt(uniform)
    # For a spiral that fills the disk uniformly, use r_i = sqrt((i + 0.5) / N)
    r = np.sqrt((indices + 0.5) / N)
    theta = golden_angle * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack([x, y])
    
    # Ensure all points are strictly inside the unit disk (with small margin)
    # The spiral should already be inside, but let's verify
    radii = np.linalg.norm(centers, axis=1)
    mask = radii < 1.0
    centers = centers[mask]
    
    # If we lost any points, add random ones inside
    if len(centers) < N:
        extra_n = N - len(centers)
        extra_r = np.sqrt(rng.random(extra_n)) * 0.95
        extra_theta = rng.random(extra_n) * 2 * np.pi
        extra_x = extra_r * np.cos(extra_theta)
        extra_y = extra_r * np.sin(extra_theta)
        extra_centers = np.column_stack([extra_x, extra_y])
        centers = np.vstack([centers, extra_centers])
    
    centers = centers[:N]
    
    # --- Step 2: Repulsion-based relaxation ---
    # We want to maximize the minimum distance. We'll use an iterative approach
    # where we compute a "score" and adjust positions to improve it.
    
    # Hyperparameters
    n_iterations = 5000
    initial_step = 0.01
    min_step = 1e-8
    decay = 0.9995
    repulsion_power = 2  # Use inverse-square-like repulsion for smoothness
    
    step_size = initial_step
    best_centers = centers.copy()
    best_score = -np.inf
    
    for it in range(n_iterations):
        # Compute pairwise distances
        # For efficiency with N=300, we can compute all pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Set diagonal to a large value to avoid division by zero
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces
        # Force on point i from point j: proportional to (1/dist^2) * (r_i - r_j) / dist
        # This gives a repulsive force that is stronger when points are close
        # We only consider pairs that are too close (below some threshold) to save time
        # But for N=300, full computation is fine
        
        # Repulsion force: F_ij = (1 / dist_ij^2) * (c_i - c_j) / dist_ij
        # This simplifies to (c_i - c_j) / dist_ij^3
        # But we want to avoid huge forces when dist is very small, so we clamp dist
        dist_clamped = np.maximum(dist, 1e-6)
        force_mag = 1.0 / (dist_clamped ** 3)
        np.fill_diagonal(force_mag, 0.0)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force: push points back inside if they're too close to the boundary
        radii = np.linalg.norm(centers, axis=1)
        boundary_margin = 0.01
        # If radius > 1 - margin, push inward
        boundary_forces = np.zeros_like(centers)
        too_close = radii > (1.0 - boundary_margin)
        if np.any(too_close):
            # Direction inward is -centers / |centers|
            inward = -centers[too_close] / radii[too_close, np.newaxis]
            # Strength proportional to how far beyond the margin
            excess = radii[too_close] - (1.0 - boundary_margin)
            boundary_forces[too_close] = inward * excess[:, np.newaxis] * 10.0
        
        total_forces = forces + boundary_forces
        
        # Update positions
        centers = centers + step_size * total_forces
        
        # Keep points inside the unit disk
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 1.0
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.999
        
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
    
    # Final cleanup: ensure all points are strictly inside
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
    
    return best_centers
# EVOLVE-BLOCK-END
