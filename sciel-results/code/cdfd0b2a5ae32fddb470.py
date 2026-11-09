"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """
    Compute the maximum possible radius for the given centers.
    Returns (min_pair_dist / 2, min_wall_dist).
    """
    # Pairwise distances
    # centers is (N, 2)
    # We need min_{i<j} |c_i - c_j|
    # Using broadcasting creates an (N, N) matrix, which is 50x50=2500 elements. Very fast.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Square is [-0.5, 0.5]^2
    # min_k (1/2 - |c_ik|) is equivalent to min(0.5 - |x|, 0.5 - |y|)
    abs_coords = np.abs(centers)
    wall_dists = 0.5 - np.max(abs_coords, axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # The radius is limited by both
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _initial_honeycomb():
    """
    Generate a honeycomb-like initial configuration.
    This is often a good starting point for circle packing.
    """
    # Approximate honeycomb structure
    # Rows of circles offset
    centers = []
    # Try to fit 50 circles
    # A 7x8 grid is 56, maybe we can prune or adjust.
    # Let's just use a jittered grid for robustness and speed in initialization
    # Honeycomb is complex to generate exactly for N=50 without knowing the specific optimal layout.
    # A jittered grid is a solid baseline.
    
    # Grid 7x8 = 56 points. We take 50.
    # Or 8x7.
    # Let's use a simple jittered grid.
    m = 8
    n = 7
    # Total points m*n = 56. We need 50.
    # We can generate all and pick best 50? Or just generate 50 directly.
    # Let's generate a dense grid and then randomize.
    
    # Actually, random initialization + SA is often better than a fixed structure if SA is strong.
    # Let's start with a random distribution inside the square.
    np.random.seed(42) # Deterministic start for reproducibility in this block, but we might want different seeds
    centers = np.random.uniform(-0.4, 0.4, size=(N, 2))
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    # Try a few different initializations
    best_centers = None
    best_radius = 0.0
    
    # Strategy: Simulated Annealing
    # Parameters
    T_start = 0.01
    T_end = 1e-5
    alpha = 0.9995  # Cooling rate
    # We need to estimate number of steps.
    # Let's run until time runs out.
    
    # Better initialization: Jittered Grid
    # 8x7 grid covers 56 positions.
    xs = np.linspace(-0.45, 0.45, 8)
    ys = np.linspace(-0.45, 0.45, 7)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Randomly select 50 points from the 56 grid points, then jitter them
    idx = np.random.choice(len(grid_centers), N, replace=False)
    centers = grid_centers[idx].copy()
    centers += np.random.uniform(-0.01, 0.01, size=(N, 2))
    
    # Clip to square
    centers = np.clip(centers, -0.49, 0.49)
    
    current_radius = _compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    T = T_start
    step_count = 0
    no_improve_count = 0
    
    # Pre-compute indices for faster updates? 
    # For N=50, full recomputation is ~2500 ops. 
    # 100,000 steps * 2500 ops = 2.5e8 ops. 
    # NumPy is fast, but Python loop overhead is the bottleneck.
    # We can batch moves? No, SA needs sequential acceptance.
    # We can speed up _compute_radius.
    
    # Optimized radius calculation
    def fast_radius(c):
        # c is (N, 2)
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff*diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Walls
        walls = 0.5 - np.max(np.abs(c), axis=1)
        min_wall = np.min(walls)
        
        return min(min_pair / 2.0, min_wall)
    
    # Re-evaluate with fast function
    current_radius = fast_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    while time.time() < deadline:
        # Pick a random index
        i = np.random.randint(N)
        
        # Propose a move
        # Step size should decrease with T
        step_size = T * 0.5 
        # Random direction
        move = np.random.normal(0, step_size, size=2)
        
        new_centers = centers.copy()
        new_centers[i] += move
        # Keep inside square? 
        # If it goes outside, radius becomes negative or small.
        # It's better to allow it and let the penalty handle it, 
        # or clip. Clipping creates a 'stick' to the wall which might help.
        # Let's clip to [-0.5, 0.5] to ensure validity.
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_radius = fast_radius(new_centers)
        
        delta = new_radius - current_radius
        
        if delta > 0:
            # Accept
            centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                centers = new_centers
                current_radius = new_radius
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cooling
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Local refinement every 1000 steps if stuck?
        # Or just let SA run.
        
        # If T is very low, we might get stuck. 
        # We can reset T occasionally if no improvement for a long time.
        if no_improve_count > 5000:
            # Reheat
            T = max(T, 0.005)
            no_improve_count = 0
            # Jitter current best?
            # centers = best_centers + np.random.normal(0, 0.01, size=(N,2))
            # current_radius = fast_radius(centers)
            pass

    return best_centers
# EVOLVE-BLOCK-END
