"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def compute_radius(centers):
    """
    Compute the maximum radius r for the given centers.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(centers)
    # Pairwise distances
    # Compute all pairwise differences
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def simulated_annealing(initial_centers, rng, time_limit=115.0):
    """
    Run Simulated Annealing to maximize the radius.
    """
    current_centers = initial_centers.copy()
    current_score = compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.05  # Initial temperature
    cooling_rate = 0.9995  # Cooling factor
    max_steps = 500000
    
    step_count = 0
    
    while T > 1e-5 and step_count < max_steps:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a perturbation
        num_moves = rng.integers(1, 3) # Move 1 or 2 particles
        indices = rng.choice(N, size=num_moves, replace=False)
        
        # Perturbation magnitude scales with T
        perturbation = rng.normal(0, T, size=(num_moves, 3))
        
        new_centers = current_centers.copy()
        new_centers[indices] += perturbation
        
        # Clip to bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = compute_radius(new_centers)
        
        # Metropolis criterion
        if new_score > current_score:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_score = new_score
        
        T *= cooling_rate
        step_count += 1
        
    return best_centers, best_score

def construct():
    """
    Use multiple Simulated Annealing runs with different initializations
    to maximize the minimum distance between sphere centers.
    """
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Time budget management
    total_time_limit = 115.0 # Leave 5s for overhead
    start_total = time.time()
    
    # Define initializations
    # 1. Random uniform
    init_random = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    # 2. Grid-based perturbed
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N].copy()
    init_grid += rng.normal(0, 0.02, size=init_grid.shape)
    init_grid = np.clip(init_grid, -0.49, 0.49)
    
    # 3. Another random with different seed logic
    init_random2 = rng.uniform(-0.45, 0.45, size=(N, 3))
    
    initializations = [init_random, init_grid, init_random2]
    
    # Run SA on each initialization
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_total
        if elapsed > total_time_limit:
            break
            
        # Allocate remaining time / remaining trials
        remaining_trials = len(initializations) - i
        time_per_trial = (total_time_limit - elapsed) / remaining_trials
        time_per_trial = min(time_per_trial, 30.0) # Cap per trial to ensure diversity
        
        centers, score = simulated_annealing(init, rng, time_limit=time_per_trial)
        
        if score > best_score:
            best_score = score
            best_centers = centers
    
    # If we have time left, do a final polish on the best solution
    if best_centers is not None:
        elapsed = time.time() - start_total
        if elapsed < total_time_limit - 5.0:
            # Run a short SA on the best solution to polish
            centers, score = simulated_annealing(best_centers, rng, time_limit=total_time_limit - elapsed - 2.0)
            if score > best_score:
                best_score = score
                best_centers = centers

    return best_centers
# EVOLVE-BLOCK-END
