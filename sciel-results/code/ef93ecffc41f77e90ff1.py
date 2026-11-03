import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def _radius(points):
    """
    Compute the maximum radius for the given points.
    Returns the minimum of:
    1. min distance to boundary
    2. min distance between pairs / 2
    """
    # Ensure points are within bounds for distance calculation to be valid
    # dist to walls
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    dx = np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5)
    dy = np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
    dist_walls = np.minimum(dx, dy)
    
    # Pairwise distances
    # Using broadcasting for simplicity, N=50 is small enough
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return np.min([np.min(dist_walls), min_pair_dist / 2.0])

def _objective(points_flat):
    """
    Objective function to minimize: negative radius.
    """
    points = points_flat.reshape(N, 2)
    r = _radius(points)
    return -r

def construct():
    start_time = time.time()
    deadline = 100.0 # Leave 20s buffer for safety
    
    best_points = None
    best_r = 0.0
    
    # 1. Generate initial points using a hexagonal grid
    # We want to place N points.
    # A hexagonal grid is denser than a square grid.
    # Let's try to generate a grid that covers the square and pick the N best positions.
    
    # Estimate spacing for hex grid
    # Area per point approx 1.0 / N
    # Hex area for side s is (3*sqrt(3)/2) * s^2. Distance between centers is sqrt(3)*s.
    # Let d be distance between centers. Area per point approx d^2 * sqrt(3)/2.
    # d^2 ~ 2/sqrt(3) * (1/N) => d ~ sqrt(2/(N*sqrt(3)))
    d = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate hex grid points
    # Hex grid: rows are shifted.
    # Vertical spacing: d * sqrt(3) / 2
    # Horizontal spacing: d
    # Row i: x = j * d + (i % 2) * d/2
    # y = i * d * sqrt(3) / 2
    
    # We need to cover [-0.5, 0.5]
    # Max index range
    max_idx = int(1.0 / d) + 2
    
    points_list = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = j * d + (i % 2) * d / 2.0
            y = i * d * np.sqrt(3.0) / 2.0
            # Shift to center
            x -= 0.0 # Centered at 0 already if range is symmetric
            y -= 0.0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points_list.append([x, y])
                
    points_arr = np.array(points_list)
    
    if len(points_arr) < N:
        # Fallback to square grid if hex generation failed or is too sparse
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points_arr = grid[:N]
    else:
        # We have more points than N. We need to select N points that are well spread.
        # Simple heuristic: pick the first N points in a sorted order (e.g., by y then x)
        # Or better: use a greedy selection to maximize min distance?
        # For simplicity and speed, let's just take a subset that covers the area well.
        # Sorting by y then x and taking every k-th point?
        # Actually, let's just take the first N points. The hex grid is regular.
        # To ensure we cover the whole square, we should probably center the grid.
        # The generation above is centered at (0,0).
        # Let's just take the N points closest to the center? No, that leaves edges empty.
        # Let's take points in a spiral from the center?
        # Or just sort by (y, x) and take N.
        
        # A better initialization: 
        # Randomly sample N points from the valid hex grid points?
        # Or, use the hex grid points, but if we have more, we can optimize the positions of N specific points.
        # Let's select N points from the hex grid. 
        # To maximize the chance of a good packing, we should pick points that are far apart.
        # Greedy farthest-point sampling on the hex grid points?
        
        # For now, let's just take a regular subset. 
        # If we have M points, and we need N, we can't just take the first N if they are clustered.
        # The hex grid generated above is a rectangular patch of the infinite hex grid.
        # Taking the first N (sorted by index i, j) will result in a rectangular block of the hex grid.
        # This is a valid starting point for local optimization.
        
        # Let's sort by y, then x to get a nice spread
        sort_idx = np.lexsort((points_arr[:, 0], points_arr[:, 1]))
        points_arr = points_arr[sort_idx]
        
        # If we have way more points, taking the first N might bias towards negative y.
        # Let's center the selection.
        if len(points_arr) > N:
            # Take the middle N points in the sorted list?
            start_idx = (len(points_arr) - N) // 2
            points_arr = points_arr[start_idx : start_idx + N]
            
        # Ensure shape is (N, 2)
        if points_arr.shape[0] != N:
             # Fallback
             m = int(np.ceil(np.sqrt(N)))
             ticks = (np.arange(m) + 0.5) / m - 0.5
             grid = np.array([(x, y) for y in ticks for x in ticks])
             points_arr = grid[:N]

    # Initial optimization
    x0 = points_arr.flatten()
    
    # Use Powell method for robustness
    # Max iterations limit
    max_iter = 2000
    
    def time_check():
        return (time.time() - start_time) > deadline

    # We will run optimization. 
    # To handle the time limit, we can't easily interrupt scipy.optimize.minimize.
    # However, N=50 is small. Powell should be fast.
    # If it's too slow, we might need to reduce max_iter or use a different method.
    
    try:
        res = minimize(_objective, x0, method='Powell', 
                       options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-10})
        
        if res.success or (time.time() - start_time) < deadline:
            best_points = res.x.reshape(N, 2)
            best_r = _radius(best_points)
            
            # Polish with Nelder-Mead for high precision
            # Nelder-Mead can be finicky with high dimensions, but for polishing it's good.
            # We only do this if we have time.
            if (time.time() - start_time) < deadline - 10:
                res2 = minimize(_objective, best_points.flatten(), method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res2.x is not None:
                    new_points = res2.x.reshape(N, 2)
                    new_r = _radius(new_points)
                    if new_r > best_r:
                        best_points = new_points
                        best_r = new_r
                        
    except Exception as e:
        # If optimization fails, use the initial points
        pass
    
    # Final check and return
    if best_points is None:
        best_points = points_arr
        
    # Ensure all points are strictly inside or on the boundary
    # The optimizer might push points slightly outside if constraints are not strictly enforced
    # The objective function penalizes going outside (radius becomes negative or small), 
    # but it doesn't hard-constrain.
    # We should clip points to the valid region to ensure validity.
    # However, clipping might break the "non-overlapping" if two points are clipped to the same boundary.
    # But the radius calculation uses the actual positions.
    # The task says "A centre on or outside the square... make the candidate invalid".
    # So we MUST ensure all centers are within [-0.5, 0.5].
    
    # Clip to valid region
    best_points = np.clip(best_points, -0.5, 0.5)
    
    # Re-calculate radius to ensure it's valid
    final_r = _radius(best_points)
    
    # If final_r is negative, it means overlap or outside. 
    # But we clipped, so outside is fixed. Overlap might still exist if the optimizer failed.
    # The score is the radius. If radius is negative, score is negative?
    # "Score: the largest common radius r ... for which the circles are disjoint and inside the square"
    # If they overlap, r is effectively 0 or negative?
    # The formula: r = min(min_dist/2, min_dist_to_wall).
    # If they overlap, min_dist is 0, so r=0.
    # So we are safe.
    
    return best_points
