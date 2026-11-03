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
    Use Physics-based Repulsive Force Relaxation to maximize the packing radius.
    This approach simulates spheres repelling each other and the walls,
    effectively pushing them to a local maximum of the minimum distance.
    """
    rng = np.random.default_rng(123)
    
    # Initialize with a jittered grid to avoid immediate collapse or symmetry locks
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = rng.choice(len(grid_points), size=N, replace=False)
    centers = grid_points[indices].copy()
    
    # Add small noise
    centers += rng.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = compute_radius(centers)
    
    # Physics parameters
    # We will iterate, calculating forces and updating positions.
    # Force is repulsive between spheres and from walls.
    n_iterations = 2000
    learning_rate = 0.005
    decay = 0.999
    
    for i in range(n_iterations):
        # Calculate pairwise differences and distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0) # Set diagonal to 1 to avoid inf/nan in force calc, then mask
        np.fill_diagonal(dists_sq, 1.0)
        
        # Repulsive force: F ~ 1/d^2 in direction of diff
        # We want to maximize min distance, so we minimize sum(1/d) or similar.
        # Let's use force proportional to 1/d^2
        # Force on i from j: (c_i - c_j) / |c_i - c_j|^3
        # This is equivalent to gradient of 1/|c_i - c_j|
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Calculate gradient components
        # dists[:, :, np.newaxis] is shape (N, N, 1)
        # diff is (N, N, 3)
        # grad = -diff / (dists[:, :, np.newaxis] ** 3)
        # We want to move in direction of -grad to minimize energy?
        # Energy E = sum 1/d. 
        # dE/dc_i = sum_j - (c_i - c_j) / |c_i - c_j|^3
        # Descent: c_i <- c_i - lr * dE/dc_i = c_i + lr * sum_j (c_i - c_j) / |c_i - c_j|^3
        
        # Compute sum of forces
        # force = diff / (dists_sq[:, :, np.newaxis] * dists[:, :, np.newaxis])
        # This is diff / dist^3
        
        dists_cubed = dists_sq * dists
        # Handle zeros (diagonal)
        dists_cubed = np.where(dists_cubed == 0, 1.0, dists_cubed)
        
        forces = diff / dists_cubed[:, :, np.newaxis]
        
        # Zero out diagonal
        forces[~mask] = 0.0
        
        # Sum forces for each particle
        net_forces = np.sum(forces, axis=1)
        
        # Update positions
        centers += learning_rate * net_forces
        
        # Clip to bounds
        centers = np.clip(centers, -0.49, 0.49)
        
        # Calculate current radius
        current_radius = compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
        # Decay learning rate
        learning_rate *= decay
        
        # Optional: Add small random noise to escape local minima
        if i % 100 == 0:
             centers += rng.normal(0, 0.001, size=centers.shape)
             centers = np.clip(centers, -0.49, 0.49)

    return best_centers
# EVOLVE-BLOCK-END
