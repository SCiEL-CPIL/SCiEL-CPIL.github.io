"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
import time

N = 200

def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    # Ensure centres are within the square [-0.5, 0.5]^2
    # Distance to walls
    wall_dist = np.minimum(
        np.minimum(
            np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
        ),
        0.5
    )
    min_wall = np.min(wall_dist)
    
    # Pairwise distances
    # For 200 points, 200*199/2 = 19900 pairs, which is manageable
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_wall, min_pair / 2.0)


def _optimize_repulsion(centres, time_limit=100.0):
    """Optimize circle packing using simultaneous repulsion and wall constraints."""
    start_time = time.time()
    n = len(centres)
    centres = centres.copy().astype(np.float64)
    
    # Compute initial radius
    best_r = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Parameters
    learning_rate = 0.001
    decay = 0.999
    repulsion_power = 4  # Stronger repulsion for closer pairs
    
    for step in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(centres)
        
        # Pairwise repulsion
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e-10)  # Avoid division by zero
        
        # Repulsion force: stronger for closer pairs
        # Use inverse power law
        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0)
        
        # Force direction
        force_mag = repulsion * learning_rate
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push circles away from walls
        # Distance to each wall
        dist_left = centres[:, 0] + 0.5
        dist_right = 0.5 - centres[:, 0]
        dist_bottom = centres[:, 1] + 0.5
        dist_top = 0.5 - centres[:, 1]
        
        # Wall forces (stronger when close to wall)
        wall_strength = 0.5
        forces[:, 0] += wall_strength * learning_rate / (dist_left ** 2 + 1e-10)
        forces[:, 0] -= wall_strength * learning_rate / (dist_right ** 2 + 1e-10)
        forces[:, 1] += wall_strength * learning_rate / (dist_bottom ** 2 + 1e-10)
        forces[:, 1] -= wall_strength * learning_rate / (dist_top ** 2 + 1e-10)
        
        # Update positions
        centres += forces
        
        # Clamp to square
        centres[:, 0] = np.clip(centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centres[:, 1] = np.clip(centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Compute current radius
        current_r = _compute_radius(centres)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # Decay learning rate
        learning_rate *= decay
    
    return best_centres, best_r


def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centres = grid[:N].copy()
    
    # Run optimization with random restarts
    best_centres = initial_centres.copy()
    best_r = _compute_radius(initial_centres)
    
    np.random.seed(42)
    num_restarts = 5
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:  # Leave 20s margin
            break
        
        # Create perturbed initial configuration
        if restart == 0:
            centres = initial_centres.copy()
        else:
            centres = initial_centres + np.random.randn(N, 2) * 0.02
        
        # Optimize
        result_centres, result_r = _optimize_repulsion(centres, time_limit=18.0)
        
        if result_r > best_r:
            best_r = result_r
            best_centres = result_centres.copy()
    
    return best_centres
