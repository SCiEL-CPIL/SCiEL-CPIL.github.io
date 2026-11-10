"""Improved program for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200

def calculate_radius(centers):
    """Calculate the radius of the packing."""
    if N < 2:
        min_dist = np.inf
    else:
        min_dist = np.min(pdist(centers, metric='euclidean'))
    
    # Min distance to boundary
    dx = np.abs(centers[:, 0])
    dy = np.abs(centers[:, 1])
    dist_boundary = np.min(np.column_stack([0.5 - dx, 0.5 - dy]), axis=1)
    min_boundary_dist = np.min(dist_boundary)
    
    radius = min(min_dist / 2.0, min_boundary_dist)
    return radius

def local_polish(centers, max_iters=100, step_size=0.001):
    """
    Perform local search to improve the minimum distance.
    Tries to move each particle in random directions to increase the minimum distance.
    """
    best_radius = calculate_radius(centers)
    best_centers = centers.copy()
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try a few random moves for particle i
            for _ in range(10):
                delta = np.random.normal(0, step_size, size=2)
                new_center = centers[i].copy()
                new_center += delta
                # Keep within bounds
                new_center[0] = np.clip(new_center[0], -0.4999, 0.4999)
                new_center[1] = np.clip(new_center[1], -0.4999, 0.4999)
                
                # Test if this move improves the minimum distance
                # We only need to check distances involving particle i and boundaries
                # But for simplicity and robustness, we check global min dist
                # Optimization: We can compute min dist to other particles and boundary for i
                # and compare with current global min.
                
                # Current global min is best_radius * 2 (if dist-limited) or best_radius (if boundary-limited)
                # Actually, best_radius is the bottleneck.
                
                # Let's just compute the new radius. It's O(N) per move, O(N^2) per iter, O(N^2 * max_iters) total.
                # 200^2 * 100 * 10 = 40,000,000 ops. Might be slow in Python loops.
                # Let's vectorize the check for particle i.
                
                # Distances from i to all others
                diffs = centers - new_center
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf
                min_dist_to_others = np.min(dists)
                
                # Distance to boundary
                d_bx = 0.5 - abs(new_center[0])
                d_by = 0.5 - abs(new_center[1])
                min_dist_to_boundary = min(d_bx, d_by)
                
                new_radius = min(min_dist_to_others / 2.0, min_dist_to_boundary)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    centers[i] = new_center
                    best_centers = centers.copy()
                    improved = True
                    break # Move to next particle
        
        if not improved:
            break
            
    return best_centers

def construct():
    """
    Construct centers of 200 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius.
    """
    best_centers = None
    best_radius = -1.0
    start_time = time.time()
    time_limit = 110.0
    
    num_restarts = 15
    steps_per_restart = 2000
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initialization: Sunflower pattern
        i = np.arange(N)
        r = np.sqrt((i + 0.5) / N)
        theta = 2 * np.pi * (i + 0.5) / golden_ratio
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale to fit well inside the square
        # Max radius of pattern is 1.0. We want it to fit in [-0.5, 0.5]
        # With some margin to allow for expansion
        scale = 0.48 / np.max(np.abs(np.column_stack([x, y])))
        x *= scale
        y *= scale
        
        centers = np.column_stack([x, y])
        # Add small jitter to break symmetry
        centers += np.random.normal(0, 0.002, size=(N, 2))
        centers = np.clip(centers, -0.499, 0.499)
        
        # 2. Force-based Simulation
        learning_rate = 0.005
        decay_rate = 0.99
        cutoff = 0.3
        temperature = 0.01
        
        current_best_radius = calculate_radius(centers)
        if current_best_radius > best_radius:
            best_radius = current_best_radius
            best_centers = centers.copy()
            
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Mask for interactions
            mask = dist_matrix < cutoff
            mask = np.where(mask, 1.0, 0.0)
            np.fill_diagonal(mask, 0.0)
            
            # Avoid division by zero
            dist_safe = np.where(dist_matrix > 1e-10, dist_matrix, 1e-10)
            
            # Force magnitude: Repulsive
            # Use 1/d^2 for stability, capped
            factor = np.zeros_like(dist_matrix)
            # Only apply force if within cutoff
            f_mag = 1.0 / (dist_safe ** 2)
            # Cap force to prevent explosions
            f_mag = np.minimum(f_mag, 10.0)
            factor[mask > 0] = f_mag[mask > 0]
            
            pair_forces = np.sum(diff * factor[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            d_right = 0.5 - centers[:, 0]
            d_left = 0.5 + centers[:, 0]
            d_top = 0.5 - centers[:, 1]
            d_bottom = 0.5 + centers[:, 1]
            
            f_x = np.zeros(N)
            f_y = np.zeros(N)
            
            # Strong repulsion from boundaries
            boundary_cutoff = 0.1
            mask_r = d_right < boundary_cutoff
            f_x[mask_r] -= 1.0 / (np.maximum(d_right[mask_r], 1e-10) ** 2)
            
            mask_l = d_left < boundary_cutoff
            f_x[mask_l] += 1.0 / (np.maximum(d_left[mask_l], 1e-10) ** 2)
            
            mask_t = d_top < boundary_cutoff
            f_y[mask_t] -= 1.0 / (np.maximum(d_top[mask_t], 1e-10) ** 2)
            
            mask_b = d_bottom < boundary_cutoff
            f_y[mask_b] += 1.0 / (np.maximum(d_bottom[mask_b], 1e-10) ** 2)
            
            boundary_forces = np.column_stack([f_x, f_y])
            
            total_forces = pair_forces + boundary_forces
            
            # Normalize forces to keep step size controlled
            force_norms = np.linalg.norm(total_forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 0:
                total_forces = (total_forces / max_norm) * learning_rate
            else:
                total_forces = np.zeros_like(centers)
                
            # Annealing Jitter
            temperature *= decay_rate
            if temperature > 1e-6:
                total_forces += np.random.normal(0, temperature, size=(N, 2))
                
            centers += total_forces
            centers = np.clip(centers, -0.4999, 0.4999)
            
            learning_rate *= decay_rate
            
            if step % 20 == 0:
                current_radius = calculate_radius(centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
                    
        # 3. Local Polish
        if time.time() - start_time < time_limit - 5:
            polished_centers = local_polish(centers, max_iters=50, step_size=0.0005)
            polished_radius = calculate_radius(polished_centers)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers.copy()
                
    return best_centers
