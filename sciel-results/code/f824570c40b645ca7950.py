"""Pack 300 equal circles in a unit disk using efficient grid-based local search."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fibonacci_disk(n, radius=1.0):
    """
    Generate n points uniformly distributed in a disk of given radius
    using the Fibonacci spiral method.
    """
    angle = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _compute_min_radius(points):
    """
    Compute the maximum possible radius for the given points.
    """
    if len(points) == 0:
        return 0.0
    
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists_to_center)
    
    if boundary_min < 0:
        return 0.0
    
    # Use a more efficient method for pairwise distances if needed, 
    # but for N=300, full matrix is ~90k ops which is fast.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    pair_min = min_pair_dist / 2.0
    
    return min(boundary_min, pair_min)

def _local_search(points, target_r, max_iters=500, lr=0.05, time_limit=10.0):
    """
    Local search to maximize minimum radius.
    Uses gradient ascent on the minimum distance metric.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    current_r = _compute_min_radius(points)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute current min radius and identify limiting pairs/points
        dists_to_center = np.linalg.norm(points, axis=1)
        boundary_dist = 1.0 - dists_to_center
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_dist = np.min(dist_matrix)
        pair_radius = min_pair_dist / 2.0
        boundary_radius = np.min(boundary_dist)
        
        current_r = min(pair_radius, boundary_radius)
        
        # If we've reached the target or better, stop
        if current_r >= target_r - 1e-10:
            break
            
        # Determine forces
        forces = np.zeros_like(points)
        
        # Boundary forces: push inward if close to boundary
        # Gradient of boundary violation: if dist > 1-r, push in
        # We want to increase min(1-|c|). 
        # If boundary is limiting, push all points inward slightly?
        # Better: only push points that are limiting.
        
        # Pair forces: push apart limiting pairs
        # Find all pairs with distance close to min_pair_dist
        # To be efficient, we can just use the gradient of the sum of squared overlaps 
        # or just push the specific min pair.
        
        # Let's use a simple repulsion model for all pairs within a cutoff
        cutoff = 2.0 * (current_r + 0.01)
        
        mask = dist_matrix < cutoff
        np.fill_diagonal(mask, False)
        
        # Compute forces from pairs
        # Force on i from j is proportional to (c_i - c_j) / d_ij^2 (repulsion)
        # We want to increase min distance.
        
        # Let's compute the gradient of the min distance directly is hard.
        # Instead, use a penalty function: sum max(0, 2r_target - d_ij)^2
        # But we are doing local search to *increase* r.
        
        # Alternative: 
        # If boundary is limiting: move all points towards center.
        # If pair is limiting: move those two points apart.
        
        if boundary_radius <= pair_radius:
            # Move points towards center
            # Only move points that are close to boundary
            close_to_boundary = dists_to_center > 1.0 - (boundary_radius + 0.01)
            for i in np.where(close_to_boundary)[0]:
                if dists_to_center[i] > 1e-6:
                    forces[i] -= points[i] / dists_to_center[i] * lr
        else:
            # Move limiting pairs apart
            # Find indices of min pair
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_idx[0], min_idx[1]
            
            d_vec = points[i] - points[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-6:
                d_unit = d_vec / d_norm
                forces[i] += d_unit * lr
                forces[j] -= d_unit * lr
            
            # Also consider other close pairs to avoid local minima
            close_pairs = np.argwhere(mask)
            for p in close_pairs:
                pi, pj = p[0], p[1]
                d_vec = points[pi] - points[pj]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-6:
                    d_unit = d_vec / d_norm
                    # Weight by how close they are to the min distance
                    weight = 1.0 / (1.0 + (dist_matrix[pi, pj] - min_pair_dist) * 100)
                    forces[pi] += d_unit * lr * weight
                    forces[pj] -= d_unit * lr * weight
        
        # Update points
        points += forces
        
        # Constrain to disk
        new_dists = np.linalg.norm(points, axis=1)
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis]) * 0.999
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialize
    points = _fibonacci_disk(N)
    current_r = _compute_min_radius(points)
    
    best_points = points.copy()
    best_r = current_r
    
    # 2. Local Search Loop
    # We iteratively try to increase the radius by doing local search
    # targeting a slightly higher radius.
    
    # Strategy: 
    # 1. Compute current r.
    # 2. Set target_r = current_r * 1.001
    # 3. Run local search to try to achieve target_r.
    # 4. If new r > best_r, update best.
    # 5. Repeat until time runs out.
    
    while time.time() < deadline:
        target_r = best_r * 1.005 # Try to increase by 0.5%
        
        # Run local search
        new_points = _local_search(best_points, target_r, max_iters=200, lr=0.02, time_limit=5.0)
        
        new_r = _compute_min_radius(new_points)
        
        if new_r > best_r + 1e-10:
            best_points = new_points
            best_r = new_r
        else:
            # If we didn't improve, we might be stuck in a local minimum.
            # We can try a small random perturbation to escape.
            # But for now, let's just stop if we can't improve with this method.
            # Or we can try a smaller step.
            # Let's try one more time with a smaller target.
            target_r = best_r * 1.001
            new_points = _local_search(best_points, target_r, max_iters=100, lr=0.01, time_limit=3.0)
            new_r = _compute_min_radius(new_points)
            if new_r > best_r + 1e-10:
                best_points = new_points
                best_r = new_r
            else:
                # Stuck
                break
    
    return best_points
# EVOLVE-BLOCK-END
