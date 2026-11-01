"""
Improved packing: Fast Lloyd's Algorithm + Local Neighborhood Polish.

Strategy:
1. Initialization: Jittered grid and random clusters.
2. Structural Optimization: Lloyd's Algorithm (Voronoi centroids) to equalize clearance.
3. Local Polish: Direct neighborhood search to maximize the minimum distance (bottleneck).
"""
import numpy as np
import time
from scipy.spatial import Voronoi, KDTree
import sys

N = 50

def _score(centers):
    """
    Compute the maximum common radius for given centers.
    Uses vectorized operations for speed.
    """
    centers = np.asarray(centers, dtype=np.float64)
    # Distance to walls: min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|) for each point
    # Actually, distance to wall is min(0.5 - x, 0.5 + x, ...) which is 0.5 - |x|
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dists)
    
    # Pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _get_bottlenecks(centers, r_current):
    """
    Identify which constraints are active (bottlenecks).
    Returns:
    - active_pairs: list of (i, j) pairs where dist < 2*r_current + epsilon
    - active_walls: list of (i, axis, sign) where point i is close to wall
    """
    centers = np.asarray(centers)
    eps = 1e-8
    threshold = 2.0 * r_current + eps
    
    # Wall bottlenecks
    active_walls = []
    for i in range(N):
        for axis in range(3):
            val = centers[i, axis]
            if val > 0.5 - r_current - eps:
                active_walls.append((i, axis, 1)) # Close to +0.5
            elif val < -0.5 + r_current + eps:
                active_walls.append((i, axis, -1)) # Close to -0.5
                
    # Pair bottlenecks
    active_pairs = []
    # Use KDTree for efficiency if N is large, but N=50 is small enough for O(N^2)
    # However, we want to find pairs close to 2*r.
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < threshold:
                active_pairs.append((i, j))
                
    return active_pairs, active_walls

def _polish_local(centers, max_iter=2000, time_limit=None):
    """
    Local search polish.
    We iteratively identify the bottleneck constraint and move points to alleviate it.
    """
    centers = centers.copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # Step size for local moves
    step_size = 1e-4
    
    for it in range(max_iter):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # 1. Identify bottlenecks
        active_pairs, active_walls = _get_bottlenecks(centers, best_r)
        
        if not active_pairs and not active_walls:
            # No active constraints? Increase r slightly? 
            # In this formulation, r is determined by min dist. 
            # If we move points, r might increase.
            # We just need to find a move that increases the min distance.
            pass
            
        # 2. Calculate a displacement vector for each point
        # Heuristic: Move away from nearest neighbors and away from walls.
        displacements = np.zeros_like(centers)
        
        # Process wall constraints
        for i, axis, sign in active_walls:
            # Move point i away from the wall (towards center)
            displacements[i, axis] -= sign * step_size * 0.5
            
        # Process pair constraints
        for i, j in active_pairs:
            vec = centers[i] - centers[j]
            d = np.linalg.norm(vec)
            if d < 1e-10:
                # Random jitter if coincident
                vec = np.random.normal(0, 1e-4, 3)
                d = np.linalg.norm(vec)
            norm_vec = vec / d
            # Move i away from j, and j away from i
            displacements[i] += norm_vec * step_size
            displacements[j] -= norm_vec * step_size
            
        # 3. Apply displacements with a damping factor
        # We want to be cautious. 
        # If we just apply displacements, we might oscillate.
        # Let's try a simple gradient ascent on the min distance.
        # The "gradient" of min distance is the sum of unit vectors from nearest neighbors.
        
        # Apply
        new_centers = centers + displacements
        
        # Clip to cube (safety)
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Evaluate
        new_r = _score(new_centers)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            # If we made progress, maybe keep step size or decrease slightly?
            # Keeping constant is safer for local search.
        else:
            # If no progress, decrease step size to settle into a local optimum
            step_size *= 0.95
            if step_size < 1e-8:
                break
                
    return best_centers, best_r

def _lloyd_iteration(centers, max_iter=50, deadline=None):
    """
    Lloyd's algorithm: iteratively move each center to the centroid of its
    Voronoi cell (clipped to the cube).
    """
    c = centers.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        try:
            vor = Voronoi(c)
        except Exception:
            break
            
        new_c = c.copy()
        
        for i in range(len(c)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                continue
                
            vertex_indices = [v for v in region if v >= 0]
            if len(vertex_indices) < 3:
                continue
                
            vertices = vor.vertices[vertex_indices]
            
            # Clip vertices to cube for centroid calculation
            # A simple mean of clipped vertices is a good approximation
            # for "center of mass" within the cube bounds.
            clipped_verts = np.clip(vertices, -0.5, 0.5)
            
            # If the cell is unbounded, the centroid might be skewed.
            # We check if the original point is inside the convex hull of the clipped vertices?
            # Simpler: Just use the mean.
            centroid = np.mean(clipped_verts, axis=0)
            
            # Damping to prevent oscillation
            new_c[i] = 0.5 * c[i] + 0.5 * centroid
            
        new_c = np.clip(new_c, -0.499, 0.499)
        
        # Check convergence
        displacement = np.max(np.linalg.norm(new_c - c, axis=1))
        c = new_c
        
        if displacement < 1e-9:
            break
            
    return c

def _initial_grid_jittered(seed=42):
    """Jittered grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _initial_random_cluster(seed=123):
    """Random initialization with some clustering to encourage Voronoi structure."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    return centers

def construct():
    t_start = time.time()
    deadline = t_start + 145  # 5s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Initializations
    inits = [
        _initial_grid_jittered(seed=42),
        _initial_grid_jittered(seed=123),
        _initial_random_cluster(seed=101),
        _initial_random_cluster(seed=202)
    ]
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
            
        # 1. Lloyd's Algorithm
        if time.time() > deadline - 15:
            continue
            
        lloyd_res = _lloyd_iteration(cur, max_iter=100, deadline=deadline - 10)
        lloyd_score = _score(lloyd_res)
        
        if lloyd_score > best_score:
            best_score = lloyd_score
            best_centers = lloyd_res.copy()
            cur = lloyd_res.copy()
        else:
            # If Lloyd didn't improve, stick with the best so far for polishing
            cur = best_centers.copy()
            
        # 2. Local Polish
        if time.time() > deadline - 5:
            continue
            
        # Allocate remaining time to polish
        time_left = deadline - 5 - time.time()
        if time_left > 1:
            polished_centers, polished_score = _polish_local(cur, max_iter=5000, time_limit=time_left)
            
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished_centers.copy()
    
    # Final Safety Check
    if best_centers is None:
        best_centers = _initial_grid_jittered()
        
    # Ensure final result is valid (inside cube)
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

if __name__ == "__main__":
    # Quick self-test
    centers = construct()
    r = _score(centers)
    print(f"Score: {r}")
