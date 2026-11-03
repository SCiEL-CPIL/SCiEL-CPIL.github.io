"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 300 equal circles in the unit disk using deterministic
    local optimization with hexagonal grid initialization.
    """
    
    # 1. Initialization: Hexagonal Grid
    # We want to place N points. A hexagonal grid is dense.
    # Generate points in a hex grid and select the first N that fit in radius 1.
    # Or generate a large grid and center it.
    
    points = []
    # Hex grid spacing. We don't know the optimal spacing yet, but we can estimate.
    # For N=300 in unit circle, area ~ pi. Area per circle ~ pi/300.
    # Side length of hex ~ sqrt(pi/300 / (sqrt(3)/2)) ~ 0.055.
    # Distance between centers in hex grid is ~ 0.055.
    # Let's generate a grid with spacing 0.05 and filter.
    
    # Better: Generate a spiral or grid and scale.
    # Let's use a simple radial grid or hex grid generation.
    
    # Generate hex grid points
    # Spacing s. 
    # Let's try s = 0.05.
    s = 0.05
    max_r = 1.1 # Slightly larger to ensure we have enough points
    points = []
    # Hex grid coordinates
    # x = s * (col + 0.5 * (row % 2))
    # y = s * sqrt(3) / 2 * row
    max_row = int(max_r / (s * np.sqrt(3) / 2)) + 2
    max_col = int(max_r / s) + 2
    
    for row in range(-max_row, max_row + 1):
        for col in range(-max_col, max_col + 1):
            x = s * (col + 0.5 * (row % 2))
            y = s * np.sqrt(3) / 2 * row
            if x*x + y*y <= max_r*max_r:
                points.append([x, y])
                
    points = np.array(points)
    
    # We need exactly N points. 
    # If we have more, we can pick the ones closest to the center? 
    # Or we can just take the first N. 
    # To ensure good distribution, let's sort by radius and take the innermost N?
    # No, we want them spread out. 
    # Let's just take a random subset or the first N? 
    # Actually, if we generate a grid that covers the circle, taking the first N might bias towards one side if the loop order is bad.
    # Let's shuffle and take N.
    
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        positions = points[indices].copy()
    else:
        # If not enough, pad with random points? Unlikely with s=0.05.
        # Area ~ pi. Grid area ~ (2*max_r)^2. 
        # Number of points ~ Area / (s^2 * sqrt(3)/2).
        # pi / (0.0025 * 0.866) ~ 3.14 / 0.00216 ~ 1450 points.
        # So we have plenty.
        positions = points[:N].copy()
        
    # Shuffle to remove any ordering bias
    np.random.seed(42)
    np.random.shuffle(positions)
    
    # 2. Optimization
    # We want to maximize r = min(min_dist, 1 - max_radius)
    # We will use an iterative repulsion scheme.
    # Force between i, j: if dist < target_dist, push apart.
    # Force to boundary: if dist_to_center > 1 - target_radius, push in.
    
    # We don't know the optimal target radius. 
    # We can try to maximize the minimum distance directly.
    # Let's use a "soft" repulsion that penalizes closeness.
    # Objective: Minimize sum of (1/d^2) or similar? 
    # Or maximize min distance.
    
    # Let's use a gradient ascent on the minimum distance.
    # This is hard to vectorize directly for the min.
    # Instead, let's use a repulsion force that is strong when close.
    # F = k / d^2. 
    # And a boundary force.
    
    # Let's try a specific target radius close to the known best: 0.053
    # And run a simulation that tries to maintain this radius.
    # If it fails (overlaps remain), we reduce r? 
    # But we want to *find* the max r.
    
    # Alternative: Run the optimization without a fixed target, just repelling, 
    # and see what min distance we get.
    # Then we can try to push it higher.
    
    # Let's use a cooling schedule for the step size.
    
    max_iterations = 20000
    # Initial step size
    step_size = 0.01
    cooling = 0.999
    
    # We will optimize for a fixed number of iterations.
    # To make it effective, we use a force model that pushes apart any pair closer than a certain distance.
    # Let's set the "desired" separation to be the current best min distance + epsilon?
    # No, let's just use a strong repulsion for all pairs.
    
    # Force: F_ij = k * (p_i - p_j) / |p_i - p_j|^3  (Coulomb-like)
    # This naturally maximizes separation.
    # Boundary: F_boundary = -k_b * (p_i / |p_i|) * (|p_i| - (1-r_est))^2 ?
    # Let's use a simple boundary repulsion: if |p_i| > 0.9, push in.
    
    k_repel = 1.0
    k_bound = 10.0
    bound_limit = 0.95 # Start pushing in when radius > 0.95
    
    for i in range(max_iterations):
        # Calculate forces
        # Vectorized pairwise
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Avoid div by zero
        dist[dist < 1e-9] = 1e-9
        
        # Force magnitude: k / dist^2
        # Direction: diff / dist
        # F = k * diff / dist^3
        # We sum over j for each i.
        
        # To avoid O(N^3) or large memory, N=300 is small enough for N^2 matrices.
        # 300x300x2 is small.
        
        # Mask diagonal
        np.fill_diagonal(dist, 1.0) # Dummy value, we'll zero out forces later
        
        # Force factor: k / dist^2
        # F_vec = diff * (k / dist^3)
        # factor = k / dist^2
        # F = factor[:, :, np.newaxis] * diff / dist[:, :, np.newaxis]
        # This is k * diff / dist^3
        
        # Let's compute forces
        # To save memory and time, we can just do:
        # inv_dist3 = 1.0 / (dist**3)
        # inv_dist3[np.eye(N, dtype=bool)] = 0.0
        # forces = np.sum((inv_dist3 * k_repel)[:, :, np.newaxis] * diff, axis=1)
        
        inv_dist3 = 1.0 / (dist**3)
        np.fill_diagonal(inv_dist3, 0.0)
        
        forces = np.sum((inv_dist3 * k_repel)[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force
        # dist_center = np.linalg.norm(positions, axis=1)
        # overflow = np.maximum(0, dist_center - bound_limit)
        # force_bound = -k_bound * overflow * positions / dist_center
        
        dist_center = np.linalg.norm(positions, axis=1)
        # We want to keep them inside 1.0. 
        # Let's use a quadratic penalty for exceeding 0.98
        limit = 0.98
        overflow = np.maximum(0, dist_center - limit)
        # Force is proportional to overflow, direction is -pos/dist
        safe_dist_c = np.where(dist_center > 1e-9, dist_center, 1.0)
        boundary_force = -k_bound * overflow / safe_dist_c
        forces += boundary_force[:, np.newaxis] * positions
        
        # Update positions
        # Simple Euler step with damping
        # We need to scale forces. The repulsion force can be very large if dist is small.
        # Let's cap the force magnitude or use a smaller step size.
        
        # Normalize forces? No, let's just use a small step size.
        # Step size cooling
        positions += forces * step_size
        
        # Keep inside unit circle (hard constraint for stability)
        # If any point goes outside, project it back?
        # This might cause issues with the physics, but prevents divergence.
        new_dist = np.linalg.norm(positions, axis=1)
        mask_out = new_dist > 1.0
        if np.any(mask_out):
            # Project back to surface? Or just clip?
            # Clipping to radius 1.0 - epsilon
            positions[mask_out] *= (1.0 - 1e-5) / new_dist[mask_out, np.newaxis]
            
        step_size *= cooling
        
        # Optional: Check progress
        # if i % 1000 == 0:
        #     d = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
        #     np.fill_diagonal(d, np.inf)
        #     min_d = np.min(d)
        #     max_r = np.max(np.linalg.norm(positions, axis=1))
        #     r = min(min_d/2, 1-max_r)
        #     print(f"Iter {i}, r={r}")
        
    # Final Check and Return
    # Calculate the actual radius achieved
    d = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    max_center_dist = np.max(np.linalg.norm(positions, axis=1))
    
    r = min(min_pair_dist / 2, 1 - max_center_dist)
    
    # If the result is bad (e.g. r < 0.04), we might need to restart or adjust parameters.
    # But for now, return the best found.
    
    return positions
# EVOLVE-BLOCK-END
