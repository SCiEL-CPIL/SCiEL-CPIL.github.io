"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_score(centers):
    """
    Fast score calculation.
    Returns the minimum radius.
    """
    # Wall constraint
    wall_dist = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    
    # Pairwise constraint
    # Using cdist is fast for N=100
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall, min_pair_dist / 2.0)

def _make_grid():
    """Create an initial grid of points."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate all combinations
    xs, ys, zs = np.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid = np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))
    return grid[:N].astype(np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer
    
    # 1. Initialize
    centers = _make_grid()
    best_centers = centers.copy()
    best_score = _fast_score(best_centers)
    
    # 2. Simulated Annealing / Local Search
    # We use a simple hill climbing with occasional random jumps (restarts)
    # to escape local minima.
    
    # Parameters
    current_temp = 0.05
    cooling_rate = 0.999
    
    # We will perform iterations until time runs out.
    # Each iteration involves moving one random sphere.
    
    # To speed up scoring, we can't easily do incremental updates without complex data structures.
    # However, N=100 is small enough that _fast_score is very fast (<1ms).
    # So we can do many iterations.
    
    while time.time() < deadline:
        # Move a random sphere
        idx = np.random.randint(N)
        old_pos = centers[idx].copy()
        
        # Step size: start large, shrink over time
        # But since we have a time limit, we can just use a fixed moderate step size
        # and rely on the number of iterations.
        # Let's use a step size that allows exploration.
        step_size = 0.02
        centers[idx] += np.random.randn(3) * step_size
        
        # Clip to cube
        centers[idx] = np.clip(centers[idx], -0.4999, 0.4999)
        
        new_score = _fast_score(centers)
        
        # Accept if better, or with probability exp(-delta/T) if worse (Metropolis)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
            # Reset temperature on improvement to allow further exploration
            current_temp = 0.05 
        else:
            # Check if we should accept a worse move to escape local minimum
            delta = best_score - new_score
            if np.random.rand() < np.exp(-delta / current_temp):
                # Accept
                pass
            else:
                # Reject
                centers[idx] = old_pos
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < 0.001:
            current_temp = 0.001

    # Final polish: try small moves on the best solution found
    if time.time() < deadline - 2.0:
        polish_centers = best_centers.copy()
        polish_score = best_score
        # Try 1000 small moves
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            idx = np.random.randint(N)
            old_pos = polish_centers[idx].copy()
            polish_centers[idx] += np.random.randn(3) * 0.005
            polish_centers[idx] = np.clip(polish_centers[idx], -0.4999, 0.4999)
            
            new_score = _fast_score(polish_centers)
            if new_score > polish_score:
                polish_score = new_score
                best_centers = polish_centers.copy()
                best_score = polish_score
            else:
                polish_centers[idx] = old_pos

    # Ensure strict validity
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
