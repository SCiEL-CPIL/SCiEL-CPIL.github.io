"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    # Distance to walls
    wall_dists = np.min(np.abs(centers), axis=1)
    
    # Pairwise distances
    # Using broadcasting for pairwise distances
    # dist[i, j] = |c_i - c_j|
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    # The radius for sphere i is half the min of its wall dist and min pair dist
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial Configuration: FCC-inspired or slightly perturbed grid
    # A 4x4x4 grid is 64 points. We take 50.
    # To make it better, we can use a body-centered or face-centered arrangement logic.
    # For now, let's start with a grid but jitter it slightly to break symmetry.
    
    m = 4 # 4x4x4 = 64
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle and take N to get a diverse starting set, or just take first N
    # Taking first N from a sorted grid creates a "block". 
    # Better to sample randomly from the grid or use a stratified random sample.
    # Let's use a stratified random sample within the cube to avoid clustering.
    # Actually, a perturbed grid is often a good start for Lloyd's.
    
    centers = grid[:N].copy()
    # Add small random noise to break degeneracy
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to inside cube
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Lloyd's Algorithm: Move each center to the centroid of its Voronoi cell
    # We approximate the Voronoi cell centroid by sampling points in the cube
    # and assigning them to the nearest center.
    
    # Number of sample points for centroid estimation
    n_samples = 20000
    sample_points = rng.uniform(-0.5, 0.5, size=(n_samples, 3))
    
    # Optimization: Pre-compute sample points once
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # 1. Assign sample points to nearest center
        # dists shape: (n_samples, N)
        diffs = sample_points[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)
        
        # 2. Compute centroids
        # Initialize sums and counts
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        # Efficient accumulation using bincount or scatter
        # For 50 centers, we can just loop or use advanced indexing
        # sums[nearest, 0] += sample_points[:, 0]
        np.add.at(sums, (nearest, 0), sample_points[:, 0])
        np.add.at(sums, (nearest, 1), sample_points[:, 1])
        np.add.at(sums, (nearest, 2), sample_points[:, 2])
        np.add.at(counts, nearest, 1)
        
        # Avoid division by zero
        valid = counts > 0
        new_centers = centers.copy()
        new_centers[valid] = sums[valid] / counts[valid, np.newaxis]
        
        # If a center has no samples, keep it (or move to center of mass of empty space, but keep is safer)
        
        # 3. Move centers towards new centroids with a step size
        # Full step can be unstable or oscillate. Use a step size < 1.
        # Also, we must ensure centers stay within the cube.
        step_size = 0.5
        centers = centers + step_size * (new_centers - centers)
        
        # Clip to cube
        centers = np.clip(centers, -0.49, 0.49)
        
        # 4. Evaluate
        current_radius = _compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
        # Optional: If we stagnate, we could jitter, but Lloyd's usually converges.
        # To escape local minima, we can add a small noise if no improvement for a while.
        # For now, keep it pure Lloyd's.
        
    return best_centers
# EVOLVE-BLOCK-END
