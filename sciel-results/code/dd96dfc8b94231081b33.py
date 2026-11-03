import numpy as np

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with a Repulsive Force Relaxation step to maximize
    the minimum distance between circles and between circles and the square boundary.
    """
    
    def evaluate(points):
        """
        Returns the minimum radius r for the given configuration.
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        if points is None or len(points) != N:
            return -np.inf
        
        # 1. Distance to walls
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # 2. Distance between pairs
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def relax(points, step_size, noise_scale):
        """
        Apply repulsive forces to all points to increase the minimum distance.
        """
        p = points.copy()
        
        # Calculate pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Calculate wall distances
        wall_dists = np.minimum.reduce([
            0.5 - p[:, 0],
            0.5 + p[:, 0],
            0.5 - p[:, 1],
            0.5 + p[:, 1]
        ])
        
        # Current minimum radius
        current_r = min(np.min(wall_dists), np.min(dists) / 2.0)
        
        # We want to push points away from each other and walls.
        # We define a "force" vector for each point.
        forces = np.zeros_like(p)
        
        # Threshold for active repulsion
        # Use a threshold slightly larger than the current minimum to focus on bottlenecks
        # but also allow some global relaxation.
        threshold = max(1.5 * current_r, 0.01)
        
        # Pairwise repulsion
        # Force from j on i is proportional to (threshold - dist) if dist < threshold
        # Direction is from j to i (so i moves away from j)
        # Vector diff is p_i - p_j. Normalized direction is (p_i - p_j) / dist.
        # Force on i = strength * (p_i - p_j) / dist
        
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        # diff is p_i - p_j (row i, col j)
        # We want force on i from j to be in direction (p_i - p_j)
        # So we use diff directly.
        directions = diff / safe_dists[:, :, np.newaxis]
        
        # Mask for active pairs
        mask = dists < threshold
        # Zero out diagonal
        mask[np.arange(N), np.arange(N)] = False
        
        # Strength = (threshold - dists)
        strengths = np.where(mask, (threshold - dists), 0.0)
        
        # Sum forces over all j for each i
        # forces[i] = sum_j (strengths[i,j] * directions[i,j])
        forces += np.sum(strengths[:, :, np.newaxis] * directions, axis=1)
        
        # Wall repulsion vectorized
        # If wall_dist < threshold, push inwards.
        
        dx_pos = 0.5 - p[:, 0]
        dx_neg = 0.5 + p[:, 0]
        dy_pos = 0.5 - p[:, 1]
        dy_neg = 0.5 + p[:, 1]
        
        # Force from right wall (x=0.5): push left (-x)
        mask_x_pos = dx_pos < threshold
        forces[:, 0] -= np.where(mask_x_pos, (threshold - dx_pos), 0.0)
        
        # Force from left wall (x=-0.5): push right (+x)
        mask_x_neg = dx_neg < threshold
        forces[:, 0] += np.where(mask_x_neg, (threshold - dx_neg), 0.0)
        
        # Force from top wall (y=0.5): push down (-y)
        mask_y_pos = dy_pos < threshold
        forces[:, 1] -= np.where(mask_y_pos, (threshold - dy_pos), 0.0)
        
        # Force from bottom wall (y=-0.5): push up (+y)
        mask_y_neg = dy_neg < threshold
        forces[:, 1] += np.where(mask_y_neg, (threshold - dy_neg), 0.0)
        
        # Add noise
        noise = np.random.normal(0, noise_scale, size=p.shape)
        
        # Update
        p += forces * step_size + noise
        
        # Project back to square
        p = np.clip(p, -0.5, 0.5)
        
        return p

    # Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    # Add small random jitter to break symmetry
    points += np.random.normal(0, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_score = evaluate(best_points)
    
    # Simulated Annealing parameters
    initial_temp = 0.05
    final_temp = 0.001
    iterations = 20000
    cooling_rate = (final_temp / initial_temp) ** (1.0 / iterations)
    
    temp = initial_temp
    step_size = 0.005
    
    import time
    start_time = time.time()
    time_limit = 110
    
    points = best_points.copy()
    current_score = best_score
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive noise scale based on temp
        noise_scale = temp * 0.1
        
        new_points = relax(points, step_size, noise_scale)
        new_score = evaluate(new_points)
        
        # Accept if better, or with probability exp((new - current) / temp)
        if new_score >= current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            prob = np.exp((new_score - current_score) / temp)
            if np.random.rand() < prob:
                points = new_points
                current_score = new_score
                
        temp *= cooling_rate
        
    return best_points

# EVOLVE-BLOCK-END
