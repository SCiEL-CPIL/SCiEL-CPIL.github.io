"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import math

def compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_{i,k} (0.5 - |c_ik|) )
    """
    # Calculate pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Calculate distance to boundaries
    abs_centers = np.abs(centers)
    dists_to_boundary = 0.5 - abs_centers
    min_boundary_dist = np.min(dists_to_boundary)
    
    # The radius is half the minimum of the two constraints
    r = min(min_pair_dist, min_boundary_dist) / 2.0
    
    return r

def construct():
    """
    Use Simulated Annealing to maximize the packing radius.
    This approach explores the configuration space by perturbing positions
    and accepting changes based on a temperature-dependent probability,
    allowing it to escape local minima.
    """
    rng = np.random.default_rng(42)
    
    # Initialization: Jittered grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = rng.choice(len(grid_points), size=N, replace=False)
    centers = grid_points[indices].copy()
    centers += rng.normal(0, 0.02, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    current_radius = compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Simulated Annealing parameters
    T_start = 0.01
    T_end = 0.0001
    n_iterations = 20000
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    for i in range(n_iterations):
        # Perturb a random subset of spheres
        # Moving 1-3 spheres at a time is often effective
        n_move = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_move, replace=False)
        
        old_centers = centers[idxs].copy()
        
        # Random step
        # Step size should be adaptive or related to T
        step_size = 0.01 * (T_start / T) ** 0.1 # Simple adaptation
        centers[idxs] += rng.normal(0, step_size, size=(n_move, 3))
        
        # Enforce boundaries
        centers[idxs] = np.clip(centers[idxs], -0.49, 0.49)
        
        new_radius = compute_radius(centers)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Always accept improvement
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp(delta / T)
            if delta < -10 * T: # Optimization: reject if too bad
                centers[idxs] = old_centers
            else:
                if rng.random() < math.exp(delta / T):
                    current_radius = new_radius
                else:
                    centers[idxs] = old_centers
        
        # Cool down
        T *= cooling_rate
        
    return best_centers
# EVOLVE-BLOCK-END
