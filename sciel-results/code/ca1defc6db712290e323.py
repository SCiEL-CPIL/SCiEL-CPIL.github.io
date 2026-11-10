import numpy as np
import time

N = 50

def _vectorized_relax(centers, steps=500, lr=0.01):
    """
    Vectorized force-directed relaxation.
    """
    centers = centers.copy()
    for _ in range(steps):
        # Calculate pairwise differences
        # diff[i, j, k] = centers[i, k] - centers[j, k]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        
        # Calculate distances
        # dist[i, j] = norm(diff[i, j, :])
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        # If dist is 0, force is 0 (handled by mask or epsilon)
        # We want force magnitude 1/d^2
        # Force vector = diff / d^3 * (1/d) ? No.
        # Force vector direction = diff / d
        # Force magnitude = 1 / d^2
        # So Force vector = diff / d^3
        # But we need to handle d=0.
        
        # Create a mask for non-zero distances
        mask = dist > 1e-8
        # Fill diagonal with 1 to avoid division by zero (will be masked out or zeroed later)
        # Actually, let's just use np.where
        # inv_dist_cubed = 1 / (dist**3) where dist > 1e-8 else 0
        inv_dist_cubed = np.zeros_like(dist)
        np.divide(1.0, dist**3, out=inv_dist_cubed, where=mask)
        
        # Force matrix: F[i, j] = diff[i, j] * inv_dist_cubed[i, j]
        forces = diff * inv_dist_cubed[:, :, np.newaxis]
        
        # Sum forces for each particle i
        # total_force[i] = sum_j forces[i, j]
        total_force = np.sum(forces, axis=1)
        
        # Wall repulsion
        # Distance to wall in each dimension
        # wall_dist[i, k] = 0.5 - |centers[i, k]|
        abs_centers = np.abs(centers)
        wall_dist = 0.5 - abs_centers
        
        # If wall_dist < threshold, add force
        # Force direction: -sign(centers)
        # Force magnitude: 1 / (wall_dist^2 + eps)
        threshold = 0.2
        wall_mask = wall_dist < threshold
        
        # Calculate wall force magnitude
        wall_force_mag = np.zeros_like(wall_dist)
        # Only calculate for masked elements
        np.divide(1.0, (wall_dist**2 + 1e-6), out=wall_force_mag, where=wall_mask)
        
        # Direction: -sign(centers)
        signs = np.sign(centers)
        # If center is 0, sign is 0, which is fine (no direction, but force mag might be high? No, if center is 0, wall_dist is 0.5, which is > 0.2, so mask is false)
        
        wall_force = -signs * wall_force_mag
        
        # Total force
        total_force += wall_force
        
        # Update positions
        centers += lr * total_force
        
        # Clip to bounds
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
    return centers

def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    best_score = 0.0
    best_centers = None
    
    num_restarts = 100
    
    for i in range(num_restarts):
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        seed = 42 + i
        rng = np.random.default_rng(seed)
        
        # Initial random points
        # Spread them out a bit initially
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        
        # Relax
        relaxed_centers = _vectorized_relax(centers, steps=200, lr=0.02)
        
        # Calculate score
        score = _compute_score(relaxed_centers)
        
        if score > best_score:
            best_score = score
            best_centers = relaxed_centers.copy()
            
    # Final polish: Try to push apart the closest pair
    if best_centers is not None and time.time() - start_time < time_limit - 2:
        # Identify closest pair
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        
        # Move them apart slightly
        # Vector from j to i
        vec = best_centers[i] - best_centers[j]
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 1e-9:
            direction = vec / norm_vec
        else:
            direction = np.array([1.0, 0.0, 0.0])
            
        # Move i in direction, j in -direction
        step = 0.001
        new_i = best_centers[i] + direction * step
        new_j = best_centers[j] - direction * step
        
        # Check if they stay in bounds
        if np.all(new_i > -0.5) and np.all(new_i < 0.5) and np.all(new_j > -0.5) and np.all(new_j < 0.5):
            candidate = best_centers.copy()
            candidate[i] = new_i
            candidate[j] = new_j
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_centers = candidate
                best_score = cand_score

    if best_centers is None:
        # Fallback: Simple grid
        # 50 points in a cube. 4x4x4 = 64 slots.
        # Use a subset of a grid.
        centers = np.zeros((N, 3))
        # Generate a 4x4x4 grid
        grid = np.array(np.meshgrid(np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4))).T.reshape(-1, 3)
        # Take first 50
        centers = grid[:50]
        return centers

    return best_centers
