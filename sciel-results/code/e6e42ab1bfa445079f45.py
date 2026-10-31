import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_r = 0.0
    best_pos = None
    
    # We will try to maximize r for equal circles
    # Initial guess for r
    r_start = 0.05
    
    # Number of restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize positions
        if restart % 2 == 0:
            # Random initialization
            pos = np.random.rand(N, 2)
        else:
            # Grid initialization (5x5 grid + 1 extra)
            # This provides a better starting point
            side = 5
            grid_pos = []
            for i in range(side):
                for j in range(side):
                    grid_pos.append([(i + 0.5) / side, (j + 0.5) / side])
            # Add one more circle, maybe in the center of a gap or random
            grid_pos.append([0.5, 0.5]) # This will overlap, but forces will resolve
            pos = np.array(grid_pos[:N])
            
        # Current radius
        r = r_start
        
        # Velocity
        vel = np.zeros((N, 2))
        
        # Parameters
        dt = 0.01
        damping = 0.9
        force_strength = 1.0
        
        # Local best for this restart
        local_best_r = r
        local_best_pos = pos.copy()
        
        # Optimization loop
        while time.time() < deadline:
            # Compute pairwise distances
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Check for overlaps
            np.fill_diagonal(dist, np.inf) # Ignore self
            overlaps = dist < (2 * r)
            
            # Check wall constraints
            wall_violation = np.any(pos < r) or np.any(pos > 1 - r)
            
            if not np.any(overlaps) and not wall_violation:
                # No overlaps and inside box
                # Increase r
                r += 0.0005
                if r > local_best_r:
                    local_best_r = r
                    local_best_pos = pos.copy()
            else:
                # Calculate forces
                forces = np.zeros((N, 2))
                
                # Pairwise repulsion
                overlap_depth = np.where(overlaps, (2 * r) - dist, 0.0)
                
                # Force direction
                safe_dist = np.where(dist > 0, dist, 1.0)
                dir_vec = diff / safe_dist[:, :, np.newaxis]
                
                # Sum forces
                forces = np.sum(overlap_depth[:, :, np.newaxis] * dir_vec, axis=1)
                forces *= force_strength
                
                # Wall forces
                wall_force = np.zeros((N, 2))
                
                # X walls
                mask_left_x = pos[:, 0] < r
                wall_force[mask_left_x, 0] += (r - pos[mask_left_x, 0])
                
                mask_right_x = pos[:, 0] > 1 - r
                wall_force[mask_right_x, 0] -= (pos[mask_right_x, 0] - (1 - r))
                
                # Y walls
                mask_left_y = pos[:, 1] < r
                wall_force[mask_left_y, 1] += (r - pos[mask_left_y, 1])
                
                mask_right_y = pos[:, 1] > 1 - r
                wall_force[mask_right_y, 1] -= (pos[mask_right_y, 1] - (1 - r))
                
                # Add wall forces
                forces += wall_force
                
                # Update velocity
                vel = vel * damping + forces * dt
                
                # Update position
                pos = pos + vel * dt
                
                # Clamp positions to [0, 1]
                pos = np.clip(pos, 0.0, 1.0)
                
                # Stability check
                max_overlap = np.max(overlap_depth)
                if max_overlap > 0.5:
                    r *= 0.9
                    vel = np.zeros((N, 2))
        
        # After loop, update global best
        if local_best_r > best_r:
            best_r = local_best_r
            best_pos = local_best_pos
            
    # Construct final array
    if best_pos is not None:
        result = np.hstack((best_pos, np.full((N, 1), best_r)))
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        
    return result
