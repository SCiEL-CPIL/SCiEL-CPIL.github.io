"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial guess: 7x7 grid (49 points) + 1 point at (-2.5/7, -2.5/7)
    # This topology is known to be close to optimal for N=50
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra_point = np.array([-2.5/7, -2.5/7])
    points = np.vstack([grid, extra_point])
    
    # Start with a radius slightly smaller than the grid spacing to ensure initial validity
    # Grid spacing is 1/7. Max radius in 7x7 grid is 1/14.
    # We want to expand beyond this.
    # Known best radius is ~0.0713. 1/14 is ~0.0714.
    # So we start at 0.07 and try to grow.
    
    target_r = 0.07
    max_r = 0.071377103865 # Best known
    
    # Force relaxation parameters
    dt = 0.01
    max_iter_relax = 2000
    tol = 1e-6
    
    while target_r < max_r and time.time() < deadline:
        # Try to relax at current target_r
        success = False
        for _ in range(max_iter_relax):
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            forces = np.zeros_like(points)
            min_dist = np.inf
            min_wall = np.inf
            
            # Pairwise repulsion
            # Use vectorized distance calculation
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs closer than 2*target_r + tol
            # To save time, we can just compute all, N=50 is small
            # Force is proportional to overlap / dist
            # F = k * (2*r - d) / d * unit_vector
            # We use a stiff spring for overlaps
            
            # Mask for overlapping pairs
            overlap_mask = dists < 2.0 * target_r
            # Avoid division by zero
            safe_dists = np.where(overlap_mask, dists, 1.0)
            
            # Overlap amount
            overlaps = np.where(overlap_mask, 2.0 * target_r - dists, 0.0)
            
            # Force magnitude per pair (scalar)
            # We want to push apart.
            # F_ij = (2*r - d) / d
            # Direction: (p_i - p_j) / d
            # So force on i from j is (2*r - d) / d^2 * (p_i - p_j)
            
            # Let's compute force vector sum
            # factor[i, j] = (2*r - d_ij) / d_ij^2 if overlapping else 0
            factors = np.zeros((N, N))
            factors[overlap_mask] = overlaps[overlap_mask] / (safe_dists[overlap_mask] ** 2)
            
            # Force on i = sum_j factors[i, j] * (p_i - p_j)
            # = p_i * sum_j factors[i, j] - sum_j factors[i, j] * p_j
            
            sum_factors = np.sum(factors, axis=1)
            weighted_pts = np.sum(factors[:, :, np.newaxis] * points[np.newaxis, :, :], axis=1)
            
            forces += points * sum_factors[:, np.newaxis] - weighted_pts
            
            # Wall forces
            # Push in if closer than target_r to wall
            # Left wall: x < -0.5 + r
            # Right wall: x > 0.5 - r
            # Bottom: y < -0.5 + r
            # Top: y > 0.5 - r
            
            # Calculate min wall distance for reporting
            dw_x_pos = 0.5 - points[:, 0]
            dw_x_neg = 0.5 + points[:, 0]
            dw_y_pos = 0.5 - points[:, 1]
            dw_y_neg = 0.5 + points[:, 1]
            
            min_wall = np.min(np.min(np.vstack([dw_x_pos, dw_x_neg, dw_y_pos, dw_y_neg]), axis=1))
            
            # Apply wall forces
            # If x < -0.5 + r, push right (+x)
            # Force = ( -0.5 + r - x )
            # If x > 0.5 - r, push left (-x)
            # Force = ( x - (0.5 - r) )
            
            # We can use a similar soft penalty or hard projection
            # Let's use hard projection for walls to ensure validity, 
            # and soft forces for pairs.
            # Actually, mixing is tricky. Let's use forces for walls too.
            
            # Wall forces:
            # Fx += ( -0.5 + r - x ) if x < -0.5 + r else 0
            # Fx += ( x - (0.5 - r) ) if x > 0.5 - r else 0
            
            wall_limit = 0.5 - target_r
            
            # Left
            mask_left = points[:, 0] < -wall_limit
            forces[mask_left, 0] += (-wall_limit - points[mask_left, 0])
            
            # Right
            mask_right = points[:, 0] > wall_limit
            forces[mask_right, 0] += (points[mask_right, 0] - wall_limit)
            
            # Bottom
            mask_bottom = points[:, 1] < -wall_limit
            forces[mask_bottom, 1] += (-wall_limit - points[mask_bottom, 1])
            
            # Top
            mask_top = points[:, 1] > wall_limit
            forces[mask_top, 1] += (points[mask_top, 1] - wall_limit)
            
            # Update positions
            points += dt * forces
            
            # Clip to box to prevent escape (safety)
            points = np.clip(points, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check convergence
            # If max force is small and no overlaps, we are good
            max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
            
            # Check min distance
            min_dist = np.min(dists)
            
            if max_force < 1e-8 and min_dist >= 2.0 * target_r - 1e-9 and min_wall >= target_r - 1e-9:
                success = True
                break
        
        if success:
            # Increase target radius
            target_r += 0.0001
        else:
            # Try smaller step or break
            # If we can't relax, maybe target_r is too high
            # We could try to reduce target_r slightly and restart, but for simplicity, break
            # Or try a different initial guess?
            # For now, break and return best found
            break
        
    # Final check and clip
    points = np.clip(points, -0.4999, 0.4999)
    
    # Calculate actual radius to ensure validity
    # r = min(min_pair_dist/2, min_wall_dist)
    # We return the points. The evaluator calculates the score.
    
    return points
# EVOLVE-BLOCK-END
