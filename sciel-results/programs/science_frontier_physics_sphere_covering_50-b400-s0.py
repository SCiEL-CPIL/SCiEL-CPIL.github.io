import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds safety margin
    
    best_score = np.inf
    best_config = None
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    initial_config = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Dense sampling grid for evaluation and optimization
    # Use a Fibonacci spiral with more points for the grid
    M = 5000
    j = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * j / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Function to calculate covering radius
    def calc_covering_radius(config):
        # config: (N, 3)
        # grid: (M, 3)
        # Compute distances
        # dist^2 = 2 - 2 * dot
        dots = grid @ config.T  # (M, N)
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    # Lloyd's Algorithm
    def lloyd_step(config, grid):
        dots = grid @ config.T
        min_idx = np.argmin(dots, axis=1) # Note: min dot = max angle, so we want max dot for min angle
        # Wait, arccos is decreasing. Min angle corresponds to Max dot product.
        max_idx = np.argmax(dots, axis=1)
        
        new_config = np.zeros_like(config)
        for i in range(N):
            mask = (max_idx == i)
            if np.sum(mask) > 0:
                center = np.mean(grid[mask], axis=0)
                norm = np.linalg.norm(center)
                if norm > 1e-10:
                    new_config[i] = center / norm
                else:
                    new_config[i] = config[i]
            else:
                # Re-randomize empty cell
                new_config[i] = np.random.randn(3)
                new_config[i] /= np.linalg.norm(new_config[i])
        return new_config
    
    # Local Refinement (Greedy move towards farthest point)
    def refine(config, grid, max_time):
        step_size = 0.1
        current_score = calc_covering_radius(config)
        
        while time.time() - start_time < max_time:
            # Find farthest point in grid
            dots = grid @ config.T
            min_dots = np.min(dots, axis=1) # Min dot = Max angle
            # Actually, we want the point with max angle, which is min dot.
            # But we also need to know which config point is nearest.
            
            # Compute min angle for each grid point
            angles = np.arccos(np.clip(np.min(dots, axis=1), -1.0, 1.0))
            max_angle_idx = np.argmax(angles)
            y_max = grid[max_angle_idx]
            
            # Find nearest config point to y_max
            dots_to_ymax = y_max @ config.T
            nearest_idx = np.argmax(dots_to_ymax)
            
            # Move config[nearest_idx] towards y_max
            # Gradient direction: projection of (y_max - config[nearest_idx]) onto tangent plane
            v = y_max - config[nearest_idx]
            # Project v onto tangent plane at config[nearest_idx]
            # Tangent plane normal is config[nearest_idx]
            v_proj = v - np.dot(v, config[nearest_idx]) * config[nearest_idx]
            
            if np.linalg.norm(v_proj) < 1e-10:
                break
                
            new_point = config[nearest_idx] + step_size * v_proj
            new_point /= np.linalg.norm(new_point)
            
            config[nearest_idx] = new_point
            
            # Check improvement
            new_score = calc_covering_radius(config)
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert if no improvement
                config[nearest_idx] = config[nearest_idx] - step_size * v_proj
                config[nearest_idx] /= np.linalg.norm(config[nearest_idx])
                # Decrease step size if stuck
                step_size *= 0.9
                if step_size < 0.001:
                    break
                    
        return config, current_score

    # Main loop: Try multiple restarts or just one good one?
    # Given time, let's do one robust run with Lloyd then Refine.
    
    config = initial_config.copy()
    
    # Phase 1: Lloyd's Algorithm
    for _ in range(50):
        if time.time() - start_time > deadline - 20: # Leave 20s for refinement
            break
        new_config = lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-6):
            break
        config = new_config
        
    # Phase 2: Refinement
    if time.time() - start_time < deadline - 5:
        config, score = refine(config, grid, deadline - 5)
        if score < best_score:
            best_score = score
            best_config = config.copy()
    else:
        score = calc_covering_radius(config)
        if score < best_score:
            best_score = score
            best_config = config.copy()
            
    return best_config
