"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_min_dist(centers):
    """
    Compute the minimum distance between any two sphere centers
    and the minimum distance from any center to the cube boundary.
    Returns the overall minimum distance (which is 2*r).
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    return min(min_pair_dist, min_boundary_dist)

def _simulated_annealing(centers, start_temp=0.05, end_temp=0.001, max_iter=5000):
    """
    Simulated Annealing optimization.
    Tries to move the most constrained point (or a random point) by a random step.
    Accepts worse solutions with probability exp(-delta/T) to escape local minima.
    """
    centers = centers.copy()
    current_dist = _compute_min_dist(centers)
    best_centers = centers.copy()
    best_dist = current_dist
    
    n = centers.shape[0]
    
    for i in range(max_iter):
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Choose a point to move
        # Strategy: Sometimes pick the worst point, sometimes random
        if np.random.rand() < 0.7:
            # Pick the point with the smallest clearance
            # Clearance is min(pair_dist, boundary_dist) for that point
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Min distance for each point to any other point
            min_pair_per_point = np.min(dists, axis=1)
            
            # Min distance to boundary for each point
            min_bound_per_point = np.min(0.5 - np.abs(centers), axis=1)
            
            # Overall clearance for each point
            clearance = np.minimum(min_pair_per_point, min_bound_per_point)
            
            # Pick the point with minimum clearance (with some randomness)
            # Use softmax-like selection or just argmin with noise
            # Let's just pick argmin for now, maybe add noise later
            idx = np.argmin(clearance)
        else:
            idx = np.random.randint(n)
        
        # Generate a random move
        # Step size decreases with temperature
        step_size = temp * 0.5
        delta = np.random.randn(3) * step_size
        
        new_centers = centers.copy()
        new_centers[idx] = centers[idx] + delta
        
        # Check bounds
        if not np.all(np.abs(new_centers) < 0.5):
            continue
            
        new_dist = _compute_min_dist(new_centers)
        
        # Acceptance criterion
        if new_dist > current_dist:
            current_dist = new_dist
            centers = new_centers
            if current_dist > best_dist:
                best_dist = current_dist
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_dist - current_dist) / temp)
            if temp > 0:
                prob = np.exp((new_dist - current_dist) / temp)
                if np.random.rand() < prob:
                    current_dist = new_dist
                    centers = new_centers
                    if current_dist > best_dist:
                        best_dist = current_dist
                        best_centers = centers.copy()
                    
    return best_centers

def _grid_init():
    """
    Initialize with a grid-like structure to provide a good starting point.
    50 points in a 3D grid.
    4x4x3 = 48 points, plus 2 more?
    Or 3x4x5 = 60 points, remove 10?
    Let's try 4x4x3 = 48 points centered, and add 2 in the center of faces?
    Actually, let's just use a random perturbation of a grid.
    """
    # 4x4x3 grid
    x = np.linspace(-0.35, 0.35, 4)
    y = np.linspace(-0.35, 0.35, 4)
    z = np.linspace(-0.35, 0.35, 3)
    
    centers = np.array([[x[i], y[j], z[k]] for i in range(4) for j in range(4) for k in range(3)])
    
    # Add 2 more points
    # Place them in the gaps
    centers = np.vstack([centers, [[0, 0, 0], [0.1, 0.1, 0.1]]])
    
    # Randomize slightly
    centers += np.random.randn(50, 3) * 0.01
    
    # Ensure inside cube
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    best_centers = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    # Time limit management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Try different initializations and SA runs
    
    # 1. Grid initialization
    centers_grid = _grid_init()
    optimized = _simulated_annealing(centers_grid, start_temp=0.05, end_temp=0.001, max_iter=5000)
    radius = _compute_min_dist(optimized) / 2.0
    if radius > best_radius:
        best_radius = radius
        best_centers = optimized.copy()
        
    if time.time() - start_time > time_limit:
        return best_centers
        
    # 2. Random initializations
    n_random_starts = 50
    
    for _ in range(n_random_starts):
        if time.time() - start_time > time_limit:
            break
            
        centers_rand = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Use SA with fewer iterations for random starts to save time
        optimized = _simulated_annealing(centers_rand, start_temp=0.03, end_temp=0.0005, max_iter=2000)
        
        radius = _compute_min_dist(optimized) / 2.0
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
            
    return best_centers

# EVOLVE-BLOCK-END
