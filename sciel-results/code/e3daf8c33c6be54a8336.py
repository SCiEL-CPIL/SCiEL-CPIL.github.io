import numpy as np
import time
import itertools

N = 16
D = 2

def normalize(points):
    # Center at origin
    points = points - np.mean(points, axis=0)
    # Scale so max distance is 1
    # Calculate pairwise distances
    # Efficient way:
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # max_dist = np.max(dists)
    # This is O(N^2). N=16, so 256 ops. Very fast.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    max_dist = np.max(dists)
    if max_dist > 1e-10:
        points = points / max_dist
    return points

def score(points):
    points = normalize(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 1e-10:
        return float('inf')
    # We normalized max_dist to 1, so ratio is 1/min_dist
    # The task asks for (max/min)^2. 
    # If max=1, ratio is 1/min_dist.
    # Score is (1/min_dist)^2 = 1/min_dist^2.
    # But the evaluator calculates raw value from the returned points.
    # The returned points should be the optimized ones.
    # The evaluator will calculate (max/min)^2 on the returned points.
    # So we just need to minimize the raw value.
    # Minimizing 1/min_dist is equivalent to minimizing (1/min_dist)^2.
    return 1.0 / min_dist

def simulated_annealing(initial_points, max_iters=5000, start_temp=1.0, cool_rate=0.999):
    current_points = initial_points.copy()
    current_score = score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    temp = start_temp
    for i in range(max_iters):
        # Perturb
        new_points = current_points + np.random.normal(0, 0.1, size=current_points.shape)
        new_score = score(new_points)
        
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_points = current_points.copy()
                best_score = current_score
        
        temp *= cool_rate
        if temp < 1e-6:
            break
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Generate diverse initial points
    # 1. 4x4 Grid
    grid = list(itertools.product(range(4), repeat=2))
    p_grid = np.array(grid, dtype=float)
    
    # 2. Hexagonal-like
    # Rows: 3, 4, 4, 3, 2
    # y-coords: 0, 1, 2, 3, 4 (scaled)
    # x-coords offset
    hex_points = []
    rows = [3, 4, 4, 3, 2]
    y_spacing = np.sqrt(3)/2
    for r, count in enumerate(rows):
        y = r * y_spacing
        # Center the row horizontally
        start_x = -(count - 1) / 2.0
        for c in range(count):
            x = start_x + c
            hex_points.append([x, y])
    p_hex = np.array(hex_points, dtype=float)
    
    # 3. Random
    p_rand = np.random.randn(16, 2)
    
    initial_sets = [p_grid, p_hex, p_rand]
    
    # Run SA for each initial set
    for init in initial_sets:
        while time.time() < deadline:
            points, sc = simulated_annealing(init, max_iters=2000)
            if sc < best_score:
                best_score = sc
                best_points = points
            
            # Check time
            if time.time() > deadline - 5.0:
                break
                
    if best_points is None:
        return p_grid
        
    return best_points
