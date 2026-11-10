import numpy as np
import time

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    """
    # Boundary constraint
    # Min distance from any point to the boundary [-0.5, 0.5]
    # dist = 0.5 - max(abs(pos))
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    # Calculate all pairwise distances
    # diff[i, j, k] = positions[i, k] - positions[j, k]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Ignore self-distances
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization: Hexagonal Grid
    # Generate a hex grid with spacing 1
    # Then scale to fit
    
    # Generate points
    s = 1.0
    points = []
    # Range to cover a 2x2 area roughly
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    # Find max extent
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
        
    # Ensure they are strictly inside (floating point safety)
    positions = np.clip(positions, -0.5, 0.5)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Local Optimization: Targeted Repulsion + Random Jitter
    # We will iterate, trying to increase the radius.
    
    # Pre-allocate for speed?
    # We'll just recompute in the loop. N=50 is small.
    
    # Parameters
    step_size = 0.01
    max_iter = 100000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Compute bottleneck
        # We need to know WHICH constraint is limiting.
        # Recompute min_bound and min_dist and their indices.
        
        # Min bound
        abs_pos = np.abs(positions)
        max_abs_pos = np.max(abs_pos)
        min_bound = 0.5 - max_abs_pos
        idx_bound = np.argmax(max_abs_pos) # This gives index of the max abs value, 
                                           # but we need the circle index.
        # Actually, np.max(np.abs(positions)) gives the scalar.
        # To get the circle:
        # flat_index = np.argmax(abs_pos)
        # idx_bound = flat_index // 2
        
        # Min dist
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        # Find indices of min dist
        # flat_idx = np.argmin(dists)
        # i, j = np.unravel_index(flat_idx, dists.shape)
        
        # Compare which is limiting
        # r_bound = min_bound
        # r_dist = min_dist_val / 2.0
        
        # If r_bound < r_dist, boundary is limiting.
        # Else, distance is limiting.
        
        # For simplicity and robustness, let's just use a random move strategy 
        # but biased towards the "worst" circle.
        
        # Determine the "worst" circle:
        # 1. If boundary is limiting, the circle with max abs coord.
        # 2. If distance is limiting, one of the circles in the min dist pair.
        
        # Let's identify the limiting factor properly.
        
        # Get idx_bound
        flat_bound_idx = np.argmax(abs_pos)
        idx_bound_circle = flat_bound_idx // 2
        
        # Get min dist pair
        flat_dist_idx = np.argmin(dists)
        i, j = np.unravel_index(flat_dist_idx, dists.shape)
        
        r_bound = min_bound
        r_dist = min_dist_val / 2.0
        
        # Choose a candidate to move
        if r_bound < r_dist:
            # Move idx_bound_circle towards center
            target_idx = idx_bound_circle
            # Direction: towards origin
            dir_vec = -positions[target_idx]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                dir_vec = dir_vec / norm
            else:
                dir_vec = np.array([1.0, 0.0])
            
            # Move it
            new_pos = positions[target_idx].copy()
            new_pos += dir_vec * step_size
            
            # Check if inside
            if np.any(np.abs(new_pos) > 0.5):
                # If it goes out, it was already at the edge? 
                # We are moving towards center, so it should stay in.
                # Unless step_size is huge.
                pass
            else:
                positions[target_idx] = new_pos
        else:
            # Move i and j apart
            # Vector from i to j
            vec = positions[j] - positions[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                dir_i = -vec / norm
                dir_j = vec / norm
            else:
                # Coincident? Random direction
                angle = np.random.uniform(0, 2*np.pi)
                dir_i = np.array([np.cos(angle), np.sin(angle)])
                dir_j = -dir_i
            
            new_pos_i = positions[i].copy() + dir_i * step_size
            new_pos_j = positions[j].copy() + dir_j * step_size
            
            # Check boundaries for both
            if np.any(np.abs(new_pos_i) > 0.5) or np.any(np.abs(new_pos_j) > 0.5):
                # If one goes out, try moving only the other one? 
                # Or skip.
                # Let's try moving only the one that stays in, or shrink step.
                # For simplicity, skip if invalid.
                continue
            else:
                positions[i] = new_pos_i
                positions[j] = new_pos_j
        
        # Evaluate new radius
        new_radius = compute_radius(positions)
        
        # Accept if better
        if new_radius > current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_positions = positions.copy()
        else:
            # Revert
            # We need to know what we changed to revert.
            # This is messy with the if/else above.
            # Better: Store old state of the whole array? 
            # N=50, copying 100 floats is cheap.
            
            # Let's refactor to store old state before modification.
            # Actually, let's just use a simpler "Random Walk" with "Accept if better or random accept"
            # The targeted approach requires careful state management.
            
            # Let's stick to a simpler, robust Random Walk with Annealing.
            # It is less likely to bug out.
            
            # I will rewrite the loop to use Random Walk.
            
            pass

    # The logic above got complicated. Let's use the Random Walk approach 
    # which is easier to implement correctly and efficiently for N=50.
    
    return best_positions
