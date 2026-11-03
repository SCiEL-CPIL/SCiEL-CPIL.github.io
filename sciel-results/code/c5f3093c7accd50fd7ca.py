"""Improved program using Voronoi-aware local search."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    # N=50 is small enough for O(N^2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _voronoi_local_search(centers, max_iter=5000, time_limit=15.0):
    """
    Local search using Voronoi cell inradii to guide movement.
    Identifies the active constraint (closest neighbor or wall) and moves away.
    """
    centers = centers.copy()
    best_r = _calculate_true_radius(centers)
    step_size = 0.05
    no_improve_count = 0
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        indices = np.random.permutation(N)
        
        for idx in indices:
            orig_center = centers[idx]
            
            # Determine constraints
            diffs = centers - orig_center
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[idx] = np.inf
            
            wall_dists = 0.5 - np.abs(orig_center)
            min_wall = np.min(wall_dists)
            min_pair = np.min(dists)
            
            current_r = min(min_wall, min_pair / 2.0)
            
            # Find the active constraint
            if min_pair / 2.0 < min_wall:
                # Constrained by a neighbor
                closest_idx = np.argmin(dists)
                direction = orig_center - centers[closest_idx]
            else:
                # Constrained by a wall
                if orig_center[0] > 0:
                    direction = np.array([-1.0, 0.0])
                elif orig_center[0] < 0:
                    direction = np.array([1.0, 0.0])
                elif orig_center[1] > 0:
                    direction = np.array([0.0, -1.0])
                else:
                    direction = np.array([0.0, 1.0])
            
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try moving in this direction with some noise
            best_new_center = orig_center
            best_local_r = current_r
            
            for _ in range(20):
                noise = np.random.normal(0, 1, 2)
                noise /= (np.linalg.norm(noise) + 1e-9)
                move_dir = direction + noise * 0.5
                move_dir /= (np.linalg.norm(move_dir) + 1e-9)
                
                new_center = orig_center + move_dir * step_size
                new_center = np.clip(new_center, -0.499, 0.499)
                
                # Evaluate new local radius
                new_diffs = centers - new_center
                new_dists = np.sqrt(np.sum(new_diffs**2, axis=1))
                new_dists[idx] = np.inf
                new_min_pair = np.min(new_dists) / 2.0
                
                new_wall_dists = 0.5 - np.abs(new_center)
                new_min_wall = np.min(new_wall_dists)
                
                new_local_r = min(new_min_pair, new_min_wall)
                
                if new_local_r > best_local_r:
                    best_local_r = new_local_r
                    best_new_center = new_center
            
            if best_local_r > current_r + 1e-10:
                centers[idx] = best_new_center
                improved = True
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.9
            if no_improve_count > 20:
                break
        else:
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.1)
            
        current_r = _calculate_true_radius(centers)
        if current_r > best_r:
            best_r = current_r
            
    return centers, best_r

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Greedy initialization
    greedy_centers = np.zeros((N, 2))
    greedy_centers[0] = [0, 0]
    for i in range(1, N):
        best_pos = None
        best_min_dist = -1
        for _ in range(200):
            p = np.random.uniform(-0.45, 0.45, 2)
            dists_to_existing = np.linalg.norm(greedy_centers[:i] - p, axis=1)
            min_dist_existing = np.min(dists_to_existing)
            dist_to_walls = 0.5 - np.abs(p)
            min_dist_wall = np.min(dist_to_walls)
            
            current_min = min(min_dist_existing, min_dist_wall)
            
            if current_min > best_min_dist:
                best_min_dist = current_min
                best_pos = p
        greedy_centers[i] = best_pos
    inits.append(greedy_centers)
    
    # 2. Hexagonal grid
    s = 0.14 
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 3. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 4. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0: 
            break
            
        centers, score = _voronoi_local_search(init, max_iter=3000, time_limit=20.0)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
