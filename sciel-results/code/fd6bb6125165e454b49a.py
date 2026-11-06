"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in [-0.5, 0.5]^3
    maximizing the minimum radius.
    
    Strategy:
    1. Initialize with a 5x5x4 grid.
    2. Use a smooth potential energy function (sum of inverse squared distances) 
       with L-BFGS-B to spread spheres out. This avoids the non-differentiability 
       issues of directly optimizing the minimum distance.
    3. Refine the configuration using a local "push" heuristic to maximize 
       the specific minimum distance (radius).
    """
    n = N
    
    # 1. Initialization: 5x5x4 Grid
    # 100 points. 5*5*4 = 100.
    nx, ny, nz = 5, 5, 4
    x = (np.arange(nx) + 0.5) / nx - 0.5
    y = (np.arange(ny) + 0.5) / ny - 0.5
    z = (np.arange(nz) + 0.5) / nz - 0.5
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # 2. Smooth Optimization
    # Objective: Minimize sum of 1/d^2 for all pairs and walls.
    # This pushes spheres apart and away from walls.
    
    def smooth_objective(params):
        c = params.reshape(n, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Avoid division by zero
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        # Sum of inverse squared distances
        # We use a power of 2 to make it smooth and sensitive to close pairs
        pair_energy = np.sum(1.0 / dists_sq)
        
        # Wall distances
        # Distance to each face
        # x: 0.5 - |x|, etc.
        wall_dists = np.min(np.stack([
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1]),
            0.5 - np.abs(c[:, 2])
        ], axis=1), axis=1)
        
        wall_dists = np.maximum(wall_dists, 1e-10)
        wall_energy = np.sum(1.0 / (wall_dists**2))
        
        return pair_energy + wall_energy

    # Minimize
    result = minimize(smooth_objective, centers.flatten(), method='L-BFGS-B',
                      bounds=[(-0.5, 0.5)] * (n * 3),
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    centers = result.x.reshape(n, 3)
    
    # 3. Local "Push" Optimization
    # Iteratively increase the minimum distance.
    
    def compute_min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = np.min(np.stack([
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1]),
            0.5 - np.abs(c[:, 2])
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    r = compute_min_radius(centers)
    
    # Local search
    step_size = 0.01
    for _ in range(100):
        # Find current min radius and critical constraints
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_r_pair = min_pair_dist / 2.0
        
        wall_dists = np.min(np.stack([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1]),
            0.5 - np.abs(centers[:, 2])
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_r_pair, min_wall_dist)
        
        if current_r <= r:
            # No improvement, try a random kick to escape local minimum
            centers += np.random.uniform(-step_size, step_size, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)
            continue
        
        r = current_r
        
        # Identify critical pairs (dist < r * 2 + epsilon)
        eps = 1e-5
        critical_pairs = (dists < 2 * r + eps)
        np.fill_diagonal(critical_pairs, False)
        
        # Identify critical walls
        critical_walls = (wall_dists < r + eps)
        
        # Apply forces
        moves = np.zeros_like(centers)
        
        # Pair forces
        for i in range(n):
            for j in range(i + 1, n):
                if critical_pairs[i, j]:
                    # Move i away from j, and j away from i
                    vec = centers[i] - centers[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        vec = vec / norm
                    else:
                        vec = np.array([1.0, 0.0, 0.0]) # arbitrary direction
                    moves[i] += vec * step_size
                    moves[j] -= vec * step_size
        
        # Wall forces
        for i in range(n):
            if critical_walls[i]:
                # Check which wall is closest
                c = centers[i]
                # x wall
                if 0.5 - c[0] < r + eps:
                    moves[i][0] -= step_size
                if -0.5 - c[0] < r + eps:
                    moves[i][0] += step_size
                if 0.5 - c[1] < r + eps:
                    moves[i][1] -= step_size
                if -0.5 - c[1] < r + eps:
                    moves[i][1] += step_size
                if 0.5 - c[2] < r + eps:
                    moves[i][2] -= step_size
                if -0.5 - c[2] < r + eps:
                    moves[i][2] += step_size
        
        centers += moves
        centers = np.clip(centers, -0.5, 0.5)
        
        # Reduce step size
        step_size *= 0.95
        
        if step_size < 1e-6:
            break
            
    # Final check
    final_r = compute_min_radius(centers)
    
    return centers
# EVOLVE-BLOCK-END
