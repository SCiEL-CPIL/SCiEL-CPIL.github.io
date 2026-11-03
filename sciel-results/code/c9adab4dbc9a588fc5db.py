import numpy as np
from scipy.spatial import cKDTree
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill

def _score_from_points(pts):
    """Calculate the maximum radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Min distance to wall
    wall_dist = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    
    # Min pairwise distance
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)  # We just need min, 1.0 is large enough for initial check
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        # Compute distances for all pairs
        d = 0.0
        min_d = np.inf
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.dot(diff, diff))
            if dist < min_d:
                min_d = dist
        min_pair = min_d
    
    r = min(wall_dist, min_pair / 2.0)
    return r if r > 0 else 0.0

def _local_refine(pts, time_limit):
    """
    Perform local optimization using simulated annealing / repulsion.
    We want to maximize the minimum distance.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_r = _score_from_points(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Temperature schedule
    T0 = 0.01
    T_end = 0.001
    alpha = 0.999
    
    # Max iterations based on time
    max_iters = 50000
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb a random point
        idx = np.random.randint(N)
        delta = np.random.normal(0, T0, 2)
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        # Check bounds
        if np.any(new_pts[idx, 0] < -0.5) or np.any(new_pts[idx, 0] > 0.5) or \
           np.any(new_pts[idx, 1] < -0.5) or np.any(new_pts[idx, 1] > 0.5):
            continue
            
        new_r = _score_from_points(new_pts)
        
        # Accept if better, or with probability exp((new_r - current_r) / T0)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        else:
            prob = np.exp((new_r - current_r) / T0)
            if np.random.random() < prob:
                current_pts = new_pts
                current_r = new_r
                
        T0 *= alpha
        if T0 < T_end:
            T0 = T_end
            
    return best_pts

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # 1. Generate initial hexagonal grid
    # Optimal hex spacing for unit square with 200 points
    # Area per point ~ 1/200 = 0.005
    # Hex area = sqrt(3)/2 * d^2 -> d ~ sqrt(2*0.005/sqrt(3)) ~ 0.076
    # Let's try to fit a hex grid that covers the square.
    
    # We will generate a slightly larger grid and crop to 200 points centered.
    # Or generate exactly 200 points in a hex pattern.
    
    # Method: Generate a hex grid that fills the square, then select the 200 closest to center.
    # This ensures symmetry and good packing density.
    
    # Hex grid parameters
    # dy = d * sqrt(3)/2
    # dx = d
    # Shift odd rows by d/2
    
    # We want d such that we get enough points.
    # Let's try d = 0.08 (approx)
    # 1.0 / 0.08 = 12.5 cols
    # 1.0 / (0.08 * sqrt(3)/2) = 1.0 / 0.0692 = 14.4 rows
    
    # Let's generate a grid with d=0.08, centered at 0.
    
    d = 0.08
    dy = d * np.sqrt(3) / 2
    dx = d
    
    # Generate points
    pts = []
    # Range of rows and cols to cover [-0.5, 0.5]
    # x from -0.5 to 0.5, step dx
    # y from -0.5 to 0.5, step dy
    
    # To be safe, generate a bit more
    max_range = 15
    for i in range(-max_range, max_range + 1):
        y = i * dy
        if y < -0.5 or y > 0.5:
            continue
        shift = (i % 2) * (dx / 2)
        for j in range(-max_range, max_range + 1):
            x = j * dx + shift
            if x < -0.5 or x > 0.5:
                continue
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # If we have more than N, select the N closest to center
    if len(pts) > N:
        dists = np.sum(pts**2, axis=1)
        indices = np.argsort(dists)[:N]
        pts = pts[indices]
    elif len(pts) < N:
        # If too few, reduce d
        d = 0.07
        dy = d * np.sqrt(3) / 2
        dx = d
        pts = []
        for i in range(-max_range, max_range + 1):
            y = i * dy
            if y < -0.5 or y > 0.5:
                continue
            shift = (i % 2) * (dx / 2)
            for j in range(-max_range, max_range + 1):
                x = j * dx + shift
                if x < -0.5 or x > 0.5:
                    continue
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.sum(pts**2, axis=1)
            indices = np.argsort(dists)[:N]
            pts = pts[indices]
            
    # Ensure we have exactly N points
    if len(pts) != N:
        # Fallback to simple grid
        rows = 14
        cols = 15
        dx_grid = 1.0 / cols
        dy_grid = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx_grid
        ys = -0.5 + (np.arange(rows) + 0.5) * dy_grid
        grid = np.array([(x, y) for y in ys for x in xs])
        pts = grid[:N]
        
    best_pts = pts
    best_r = _score_from_points(best_pts)
    
    # 2. Local refinement
    time_left = _TIME_LIMIT - (time.time() - t0)
    if time_left > 5:
        best_pts = _local_refine(best_pts, time_left)
        best_r = _score_from_points(best_pts)
        
    return best_pts
