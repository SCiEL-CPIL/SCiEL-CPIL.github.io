import numpy as np
import time

N = 100

def _compute_score(centers):
    """
    Compute the maximum radius for the given centers.
    Returns (radius, min_pair_idx, min_wall_idx) or just radius for simplicity in inner loop.
    For optimization, we just need the value.
    """
    # Pairwise distances
    # centers shape (N, 3)
    # dists shape (N, N)
    # Using broadcasting: (N, 1, 3) - (1, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # centers shape (N, 3)
    # dist to wall for each dim: 0.5 - abs(center)
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _get_active_points(centers, radius, tol=1e-6):
    """
    Identify points that are close to the limiting radius.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Points involved in pairs with distance < 2*radius + tol
    active_pairs = dists < (2.0 * radius + tol)
    
    # Points close to walls
    wall_dists = 0.5 - np.abs(centers)
    active_walls = (wall_dists < (radius + tol)).any(axis=1)
    
    active_indices = np.unique(np.concatenate([
        np.where(active_pairs)[0],
        np.where(active_pairs)[1],
        np.where(active_walls)[0]
    ]))
    
    return active_indices

def construct():
    # Initialize with 5x5x5 grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_radius = _compute_score(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Local Search Parameters
    step_size = 0.01
    max_steps_per_iter = 50
    kick_interval = 1000 # Add noise every N iterations if no improvement
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        current_radius = _compute_score(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck for a while, kick
        if no_improve_count > kick_interval:
            centers += np.random.normal(0, 0.05, size=(N, 3))
            # Clip to cube to ensure validity
            centers = np.clip(centers, -0.49, 0.49)
            no_improve_count = 0
            continue
            
        # Find active points
        active_indices = _get_active_points(centers, current_radius)
        
        if len(active_indices) == 0:
            # Should not happen if radius is correct, but handle just in case
            break
            
        # Try to move one active point
        idx = active_indices[np.random.randint(len(active_indices))]
        
        # Generate random direction
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        # Try moving in direction and opposite direction
        for sign in [1, -1]:
            new_centers = centers.copy()
            new_centers[idx] += sign * step_size * direction
            
            # Ensure points stay inside the cube (with small margin to avoid numerical issues)
            # Actually, the score function handles walls, but we shouldn't let them fly out
            # The score will drop if they go outside, so it's safe, but clipping helps stability
            # Let's not clip inside the loop to allow the optimizer to "want" to move in, 
            # but ensure we don't produce NaNs.
            
            new_radius = _compute_score(new_centers)
            
            if new_radius > current_radius:
                centers = new_centers
                current_radius = new_radius
                break # Move to next point/iteration
        
        iteration += 1
        
        # Adaptive step size?
        # If we improved a lot, maybe keep step size. If not, reduce?
        # For simplicity, keep fixed or slight decay.
        if iteration % 100 == 0:
            step_size *= 0.95
            if step_size < 1e-5:
                step_size = 0.01 # Reset if too small, rely on kicks

    return best_centers

# EVOLVE-BLOCK-END
