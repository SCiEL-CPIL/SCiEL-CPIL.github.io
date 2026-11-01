"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Calculate the radius of the packing."""
    # Distance to walls
    # Centers are in [-0.5, 0.5]
    # Distance to wall is min(|c|, 0.5 - |c|) for each coord
    # Actually, distance to nearest wall for center c is min(0.5 - c_x, c_x + 0.5, ...)
    # Since c is in [-0.5, 0.5], dist to wall x is min(c_x + 0.5, 0.5 - c_x)
    # This is equivalent to 0.5 - abs(c_x)
    wall_dists = np.min(np.stack([0.5 - np.abs(centers[:, 0]), 
                                  0.5 - np.abs(centers[:, 1]), 
                                  0.5 - np.abs(centers[:, 2])], axis=1), axis=1)
    
    # Distance between spheres
    # Use cdist or manual broadcasting
    # For N=50, broadcasting is fine
    # dist[i, j] = ||c_i - c_j||
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1)
    
    # The radius is half the minimum distance between any pair, and also limited by wall distance
    # r = min( min_{i<j} d_ij / 2, min_i d_i_wall )
    r_pair = np.min(min_pair_dist) / 2.0
    r_wall = np.min(wall_dists)
    return min(r_pair, r_wall)

def _optimize(centers, time_limit):
    """Local optimization using repulsion forces."""
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Step size
    step = 0.01
    decay = 0.999
    min_step = 0.0001
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            c_i = centers[i]
            # Calculate distances to all other spheres
            dists = np.sqrt(np.sum((centers - c_i)**2, axis=1))
            dists[i] = np.inf
            
            # Calculate distances to walls
            d_wall_x = min(0.5 - c_i[0], c_i[0] + 0.5)
            d_wall_y = min(0.5 - c_i[1], c_i[1] + 0.5)
            d_wall_z = min(0.5 - c_i[2], c_i[2] + 0.5)
            
            # Find the minimum constraint
            min_d = np.min(dists)
            argmin_d = np.argmin(dists)
            
            # Check if wall is closer
            wall_min = min(d_wall_x, d_wall_y, d_wall_z)
            
            force = np.zeros(3)
            
            if wall_min < min_d:
                # Push away from wall
                if d_wall_x == wall_min:
                    # Nearest wall is x=0.5 or x=-0.5
                    if c_i[0] > 0:
                        force[0] = -1.0 # Push towards -x? No, away from wall.
                        # If c_i[0] > 0, closest wall is x=0.5. Push towards -x.
                        # Wait, if c_i[0] is close to 0.5, dist is 0.5-c_i[0].
                        # Gradient of dist wrt c_i[0] is -1.
                        # We want to INCREASE dist. So move in direction of gradient?
                        # d(dist)/dc = -1. So to increase dist, we should move in direction -1 * -1 = 1?
                        # Let's check: c=0.49, dist=0.01. Move to 0.48, dist=0.02. Increased.
                        # So if c>0, move towards 0.
                        # If c<0, move towards 0.
                        # So force is always towards 0.
                        force[0] = -np.sign(c_i[0])
                    else:
                        force[0] = -np.sign(c_i[0])
                elif d_wall_y == wall_min:
                    force[1] = -np.sign(c_i[1])
                else:
                    force[2] = -np.sign(c_i[2])
            else:
                # Push away from sphere argmin_d
                c_j = centers[argmin_d]
                diff = c_i - c_j
                norm = np.linalg.norm(diff)
                if norm > 1e-9:
                    force = diff / norm
            
            # Normalize force
            norm_f = np.linalg.norm(force)
            if norm_f > 1e-9:
                force = force / norm_f
            
            # Move
            new_c = c_i + step * force
            # Clamp to cube
            new_c = np.clip(new_c, -0.5, 0.5)
            
            # Check if this move improves the local minimum for this sphere?
            # Or just accept and check global?
            # Let's check if the move improves the minimum distance involving this sphere.
            # Actually, simpler: just do the move, and check global radius at the end of the full pass.
            # But if we make a bad move, we might degrade.
            # Let's try: only accept if the minimum distance for sphere i increases.
            
            # Calculate new min distance for i
            new_dists = np.sqrt(np.sum((centers - new_c)**2, axis=1))
            new_dists[i] = np.inf
            new_min_d_pair = np.min(new_dists)
            
            new_d_wall_x = min(0.5 - new_c[0], new_c[0] + 0.5)
            new_d_wall_y = min(0.5 - new_c[1], new_c[1] + 0.5)
            new_d_wall_z = min(0.5 - new_c[2], new_c[2] + 0.5)
            new_min_d_wall = min(new_d_wall_x, new_d_wall_y, new_d_wall_z)
            
            new_local_min = min(new_min_d_pair, new_min_d_wall)
            old_local_min = min(min_d, wall_min)
            
            if new_local_min > old_local_min:
                centers[i] = new_c
                improved = True
            else:
                # Try smaller step? Or just skip?
                # For now, skip.
                pass
        
        # Update best
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Decay step
        step *= decay
        if step < min_step:
            # Reset step and add noise?
            step = 0.005
            # Add small noise to escape local minima
            noise = np.random.normal(0, 0.001, centers.shape)
            centers = np.clip(centers + noise, -0.5, 0.5)

    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initial guess: 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Try multiple random subsets or perturbations
    # We have ~110 seconds. Let's do several restarts.
    # Each restart: initialize, optimize for ~10s.
    
    num_restarts = 10
    time_per_restart = 10.0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Option 1: Random subset of grid
        # Option 2: Perturbed grid
        # Option 3: Random positions
        
        if r == 0:
            # First try: Grid
            centers = grid[:N].copy()
        elif r == 1:
            # Second try: BCC-like
            # Generate BCC points
            # Cell size 0.5
            # Points at (0,0,0) and (0.5,0.5,0.5) relative to cell
            # Let's just use random init for others
            centers = np.random.uniform(-0.5, 0.5, (N, 3))
        else:
            # Perturbed grid
            centers = grid[:N].copy()
            centers += np.random.normal(0, 0.05, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)
        
        # Optimize
        opt_centers, opt_radius = _optimize(centers, time_per_restart)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
        # If we found the known best, we can stop early?
        # Known best is ~0.13595
        if best_radius > 0.1359:
            break

    # Final check
    if best_centers is None:
        # Fallback to simple grid
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
