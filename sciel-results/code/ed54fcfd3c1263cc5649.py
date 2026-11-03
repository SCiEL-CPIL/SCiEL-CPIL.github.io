import numpy as np
import time
import math

N = 50

def compute_radius(centers):
    """
    Computes the maximum radius for the given centers.
    centers: (N, 2) array
    Returns: float
    """
    # 1. Distance to walls
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # dist to left wall: x - (-0.5) = x + 0.5
    # dist to right wall: 0.5 - x
    # dist to bottom wall: y + 0.5
    # dist to top wall: 0.5 - y
    wall_dists = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_wall = np.min(wall_dists)
    
    # 2. Distance to other circles
    # We need min distance between any pair / 2
    # N=50, so pairwise distance matrix is 50x50
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_pair_radius = min_pair_dist / 2.0
    
    return min(min_wall, min_pair_radius)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for the 120s limit
    
    # Initial configuration: Grid
    # 7x7 grid covers 49. We need 50.
    # Let's try a 7x7 grid plus one in the center of a cell? 
    # Or just 8x6 grid? 
    # Actually, a randomized grid is safer for SA.
    
    # Generate a grid of 7x7 (49 points) and add 1 point randomly nearby
    # Or better: generate 50 points on a 7x7 grid by splitting one cell?
    # Let's just use a 7x7 grid and duplicate one point with jitter, or use an 8x6 grid.
    # 8*6 = 48. Need 2 more.
    # Let's stick to a simple perturbed grid of 50 points.
    
    # Create a 7x7 grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y) for y in ticks for x in ticks])
    
    # We need 50 points. We have 49.
    # Add the 50th point at the center of the square? Or random?
    # The center (0,0) is already occupied by the grid point (3,3) if m=7 (index 3 is 0.0).
    # Let's place the 50th point randomly inside the square, or jitter one of the grid points.
    # Better: Use a 7x7 grid and randomly select one point to split into two?
    # Simplest: Just use a 7x7 grid and add a random point, then let SA fix it.
    
    centers = grid_points.copy()
    centers = np.vstack([centers, np.array([0.0, 0.0])]) # This will overlap initially, SA will fix it
    # Add small random noise to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=(N, 2))
    
    # Ensure inside bounds initially? SA handles it, but starting inside is better.
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = compute_radius(best_centers)
    
    # Simulated Annealing
    T = 0.01 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    step_size = 0.05
    
    current_radius = best_radius
    current_centers = best_centers.copy()
    
    while time.time() < deadline:
        # Select a random circle
        i = np.random.randint(N)
        
        # Generate a random move
        # Step size can be adaptive, but fixed with cooling T is standard
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_centers = current_centers.copy()
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        
        new_radius = compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_radius = new_radius
            current_centers = new_centers
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Metropolis criterion
            # Probability of accepting worse solution
            # delta = new_radius - current_radius (negative)
            # exp(delta / T)
            # If T is very small, this is 0.
            if T > T_min:
                prob = math.exp((new_radius - current_radius) / T)
                if np.random.random() < prob:
                    current_radius = new_radius
                    current_centers = new_centers
        
        # Cool
        T *= alpha
        
        # If T is too low, reheat or restart?
        # For now, just let it run. With N=50, we can do many iterations.
        # To prevent stagnation, we can occasionally reheat or randomize step size.
        # Let's add a simple restart logic: if no improvement for a long time, reheat.
        # But tracking "long time" is complex. Simple cooling usually works for small N.
        
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# We need to wrap it in the required format.
# The task asks to rewrite the program.
# I will place the logic inside the EVOLVE-BLOCK.

import numpy as np
import time
import math

N = 50

def _compute_radius(centers):
    """
    Computes the maximum radius for the given centers.
    centers: (N, 2) array
    Returns: float
    """
    # 1. Distance to walls
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    wall_dists = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_wall = np.min(wall_dists)
    
    # 2. Distance to other circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_pair_radius = min_pair_dist / 2.0
    
    return min(min_wall, min_pair_radius)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for the 120s limit
    
    # Initial configuration: Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y) for y in ticks for x in ticks])
    
    centers = grid_points.copy()
    centers = np.vstack([centers, np.array([0.0, 0.0])])
    centers += np.random.uniform(-0.01, 0.01, size=(N, 2))
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Simulated Annealing
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    step_size = 0.05
    
    current_radius = best_radius
    current_centers = best_centers.copy()
    
    while time.time() < deadline:
        i = np.random.randint(N)
        
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_centers = current_centers.copy()
        new_centers[i, 0] += dx
        new_centers[i, 1] += dy
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            current_radius = new_radius
            current_centers = new_centers
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            if T > T_min:
                prob = math.exp((new_radius - current_radius) / T)
                if np.random.random() < prob:
                    current_radius = new_radius
                    current_centers = new_centers
        
        T *= alpha
        
    return best_centers
# EVOLVE-BLOCK-END
