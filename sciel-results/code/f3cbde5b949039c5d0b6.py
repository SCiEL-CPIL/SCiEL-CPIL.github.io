"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    """
    Strategy: Discrete Lattice Search.
    1. Define a fine grid of candidate points inside the cube.
    2. Use a Greedy MaxMin algorithm to select N points.
    3. Refine using local search on the grid (try moving each point to neighbors).
    4. Return the best configuration found within time limit.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Grid resolution. 
    # Too fine -> too many points, slow greedy.
    # Too coarse -> limited positions.
    # Let's try grid_size = 25. 25^3 = 15625 points.
    # Or 30^3 = 27000.
    # Let's stick to 25 for speed in Greedy, then maybe refine with a finer local grid?
    # Actually, let's use a grid that is a multiple of the cube size.
    # Cube is [-0.5, 0.5].
    # Grid step h = 1/G. Points at -0.5 + i*h.
    # To ensure spheres fit, we don't need to restrict grid points to be far from boundary 
    # because the radius calculation handles the boundary constraint. 
    # However, placing a center too close to the boundary reduces the max possible radius 
    # for that sphere. The MaxMin criterion will naturally avoid bad boundary placements 
    # if interior placements offer better min-dist.
    
    G = 25
    h = 1.0 / G
    # Points: -0.5, -0.5+h, ..., 0.5-h
    # Wait, if G=25, points are -0.5, -0.46, ..., 0.46. 
    # The last point is -0.5 + 24*(1/25) = -0.5 + 0.96 = 0.46.
    # We miss the boundary 0.5. 
    # It's better to have points slightly inset or just use the grid.
    # Let's use np.linspace(-0.5, 0.5, G)
    coords = np.linspace(-0.5, 0.5, G)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    M = len(grid)
    
    # Precompute squared norms of grid points for fast distance calc
    grid_norms = np.sum(grid**2, axis=1)
    
    # Greedy MaxMin Initialization
    selected_indices = []
    selected_points = []
    min_dists = np.full(M, np.inf) # Minimum distance to any selected point
    
    for k in range(N):
        if time.time() - start_time > time_limit - 5.0:
            break # Safety break
            
        if k == 0:
            # First sphere: center of cube? Or any point? 
            # MaxMin from empty set is undefined. 
            # Symmetry suggests center.
            # Find index closest to [0,0,0]
            idx = np.argmin(np.sum(grid**2, axis=1))
        else:
            # Find the point with the maximum min_dist
            idx = np.argmax(min_dists)
            
        selected_indices.append(idx)
        selected_points.append(grid[idx])
        
        # Update min_dists for all grid points
        # dist^2 = grid_norms - 2 * grid @ p + p_norm
        p = grid[idx]
        p_norm = np.dot(p, p)
        dists = grid_norms - 2.0 * (grid @ p) + p_norm
        min_dists = np.minimum(min_dists, dists)
        
    if len(selected_points) < N:
        # Fill remaining with random or just return what we have? 
        # Must return N points. 
        # If we broke early, we have fewer. 
        # Let's just pad with center (will be invalid/overlap, but we need shape N,3)
        # Better: don't break early in greedy, it's fast.
        pass

    centers = np.array(selected_points)
    
    # Local Search on Grid
    # For each sphere, try moving to neighbors in the grid.
    # Neighbors: indices that differ by 1 in grid index space.
    # We need a mapping from grid point to its neighbors.
    # Grid is GxGxG.
    # Point index i corresponds to (x, y, z) where i = z*G*G + y*G + x.
    
    # Precompute neighbor lists? 
    # For M=15625, storing neighbors is fine.
    # But checking all neighbors for all spheres in each iteration is:
    # N * 26 * (distance calc to other N-1 spheres)
    # 50 * 26 * 49 * 3 ops ~ 200k ops per iteration. Very fast.
    
    def compute_radius(centers):
        # Min distance between pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Min distance to boundary
        # Boundary is at +/- 0.5
        # dist to boundary = 0.5 - max(|x|, |y|, |z|) for each point? 
        # No, dist to face. 
        # min_k (0.5 - |c_ik|)
        # c_ik is coord k. 
        # dist to boundary for point i is min(0.5 - x_i, 0.5 - y_i, 0.5 - z_i, 0.5 + x_i, ...)
        # Which is 0.5 - max(|x_i|, |y_i|, |z_i|)
        max_abs = np.max(np.abs(centers), axis=1)
        min_boundary_dist = np.min(0.5 - max_abs)
        
        return min(min_pair_dist, 2.0 * min_boundary_dist) / 2.0
        # Wait, radius r = min( min_pair_dist/2, min_boundary_dist )
        # My formula: min(min_pair_dist, 2*min_boundary_dist)/2 = min(min_pair_dist/2, min_boundary_dist). Correct.

    current_radius = compute_radius(centers)
    
    # Mapping from flat index to (x, y, z) grid coords
    # i = z*G*G + y*G + x
    def idx_to_xyz(i):
        x = i % G
        y = (i // G) % G
        z = i // (G*G)
        return x, y, z

    def xyz_to_idx(x, y, z):
        if 0 <= x < G and 0 <= y < G and 0 <= z < G:
            return z*G*G + y*G + x
        return -1

    improved = True
    while improved and (time.time() - start_time < time_limit - 2.0):
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit - 2.0:
                break
            cx, cy, cz = idx_to_xyz(selected_indices[i])
            
            # Generate neighbors
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        new_idx = xyz_to_idx(cx+dx, cy+dy, cz+dz)
                        if new_idx != -1:
                            # Avoid moving to a spot already occupied by another sphere
                            # (Check if new_idx is in selected_indices and not i)
                            if new_idx not in selected_indices or new_idx == selected_indices[i]:
                                neighbors.append(new_idx)
            
            for n_idx in neighbors:
                # Try moving sphere i to n_idx
                new_centers = centers.copy()
                new_centers[i] = grid[n_idx]
                
                # Quick check: is it better?
                # Calculate new radius
                # Optimization: Only check distances involving sphere i
                # 1. Distances from i to others
                # 2. Boundary distance for i
                # 3. Min pair distance among others (unchanged)
                
                # Get old min pair dist among others
                # This is expensive to recompute fully every time.
                # Let's just use full compute_radius for simplicity first, 
                # as N=50 is small.
                
                new_r = compute_radius(new_centers)
                if new_r > current_radius + 1e-12:
                    centers = new_centers
                    selected_indices[i] = n_idx
                    current_radius = new_r
                    improved = True
                    # Break out of neighbor loop? 
                    # Greedy: take first improvement.
                    break
    
    return centers

# EVOLVE-BLOCK-END
