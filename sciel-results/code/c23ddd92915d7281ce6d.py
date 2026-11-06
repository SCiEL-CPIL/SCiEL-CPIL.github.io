"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # 1. Initialize with a grid
    # 5x5x4 grid = 100 points
    # Dimensions: 5, 5, 4
    nx, ny, nz = 5, 5, 4
    # Create grid points
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if len(grid) < N:
        # Fallback if grid size is wrong (should not happen for N=100)
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
    else:
        grid = grid[:N]
        
    best_centers = grid.copy()
    best_radius = compute_radius(best_centers)
    
    # 2. Local Search with Annealing/Shaking
    # We will run multiple iterations of local search.
    # Each iteration:
    #   - Start from a perturbed version of the best known solution.
    #   - Perform gradient-like steps to increase the minimum distance.
    
    num_restarts = 20 # Adjust based on time
    steps_per_restart = 5000
    
    # Precompute indices for pairs? 
    # For N=100, pairs are 4950. This is small enough to compute on the fly or vectorize.
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize for this restart
        if restart == 0:
            centers = best_centers.copy()
        else:
            # Perturb best centers
            # Use a small random perturbation
            centers = best_centers + np.random.normal(0, 0.01, (N, 3))
            # Clip to valid range to ensure validity
            centers = np.clip(centers, -0.49, 0.49)
            
        # Local Search Loop
        current_radius = compute_radius(centers)
        step_size = 0.005
        min_step_size = 1e-6
        
        for step in range(steps_per_restart):
            if time.time() - start_time > timeout:
                break
            
            # Compute all pairwise distances
            # Use broadcasting for efficiency
            # dists[i, j] = |c_i - c_j|
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Find the minimum pair distance
            # We need to ignore diagonal (i=j)
            np.fill_diagonal(dists, np.inf)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_pair_dist = dists[min_pair_idx]
            
            # Compute wall distances
            # dist to wall in x: 0.5 - |x|
            wall_dists = 0.5 - np.abs(centers)
            # min wall dist for each sphere
            min_wall_per_sphere = np.min(wall_dists, axis=1)
            min_wall_idx = np.argmin(min_wall_per_sphere)
            min_wall_dist = min_wall_per_sphere[min_wall_idx]
            
            current_radius = min(min_pair_dist, min_wall_dist)
            
            # Update best
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                
            # Determine move
            if min_pair_dist < min_wall_dist:
                # Move pair apart
                i, j = min_pair_idx
                if i > j: i, j = j, i # Ensure i < j for consistent indexing if needed, not strictly required
                v = centers[i] - centers[j]
                if min_pair_dist > 1e-9:
                    v = v / min_pair_dist
                else:
                    v = np.random.normal(0, 1, 3)
                    v /= np.linalg.norm(v)
                
                # Move i in direction v, j in -v
                centers[i] += step_size * v
                centers[j] -= step_size * v
            else:
                # Move sphere towards center
                k = min_wall_idx
                # Determine which wall is closest
                # wall_dists[k] has shape (3,)
                # Find the dimension with min wall dist
                dim = np.argmin(wall_dists[k])
                # Move towards center (0) in that dimension
                # If centers[k, dim] > 0, move negative. If < 0, move positive.
                move_dir = -np.sign(centers[k, dim])
                if move_dir == 0:
                    move_dir = 1.0
                centers[k, dim] += step_size * move_dir
                
            # Clip to ensure we don't go outside (though we are moving inwards or apart, 
            # moving apart might push one out if not careful, but we clip)
            centers = np.clip(centers, -0.5, 0.5)
            
            # Anneal step size
            step_size *= 0.999
            if step_size < min_step_size:
                break

    return best_centers

def compute_radius(centers):
    """
    Computes the largest common radius for the given centers.
    """
    if len(centers) < 2:
        return 0.5 - np.max(np.abs(centers))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

# EVOLVE-BLOCK-END
