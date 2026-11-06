"""
Strategy: Constructive Geometric Heuristics + Deterministic Area-Aware Local Search
This program abandons generic stochastic search for a hybrid approach:
1. Generate structured initial configurations based on geometric principles 
   (affine grids, parallel line clusters, star patterns) that inherently 
   produce many equal-area triangles.
2. Apply a deterministic local search that uses "repair" moves: for a given 
   target area T, if a point is not contributing to many T-triangles, try 
   moving it to a location that maximizes the number of new T-triangles 
   formed with existing pairs.
3. Use a fast vectorized scoring function to evaluate candidates.
"""

import numpy as np
import time
import random
from collections import Counter

def _compute_score_fast(points):
    """
    Compute the maximum multiplicity of any non-zero doubled area.
    points: numpy array of shape (30, 2)
    Returns: (max_multiplicity, best_area)
    """
    n = len(points)
    if n < 3:
        return 0, 0
    
    # Extract coordinates
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Precompute pairwise differences? No, direct triple loop is fine for n=30
    # But we can vectorize the inner loop slightly or just use a tight loop.
    # For n=30, 30*29*28/6 = 4060 triples.
    
    areas = np.zeros(4060, dtype=np.int64)
    idx = 0
    for i in range(n - 2):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n - 1):
            xj, yj = xs[j], ys[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            # Vectorize over k
            # k ranges from j+1 to n-1
            xk = xs[j+1:]
            yk = ys[j+1:]
            # det = dx_ij * (yk - yi) - dy_ij * (xk - xi)
            dets = dx_ij * (yk - yi) - dy_ij * (xk - xi)
            # Filter non-zero
            abs_dets = np.abs(dets)
            non_zero = abs_dets[abs_dets > 0]
            if len(non_zero) > 0:
                areas[idx:idx+len(non_zero)] = non_zero
                idx += len(non_zero)
    
    if idx == 0:
        return 0, 0
        
    # Count frequencies
    # Using np.unique with return_counts is fast
    unique_areas, counts = np.unique(areas[:idx], return_counts=True)
    max_count = np.max(counts)
    best_area = unique_areas[np.argmax(counts)]
    
    return int(max_count), int(best_area)

def _count_specific_area(points, T):
    """
    Count the number of triples with doubled area exactly T.
    """
    n = len(points)
    xs = points[:, 0]
    ys = points[:, 1]
    count = 0
    for i in range(n - 2):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n - 1):
            xj, yj = xs[j], ys[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            xk = xs[j+1:]
            yk = ys[j+1:]
            dets = dx_ij * (yk - yi) - dy_ij * (xk - xi)
            # Count where abs(det) == T
            count += np.sum(np.abs(dets) == T)
    return count

def _generate_initial_configs():
    """
    Generate a variety of structured initial configurations.
    """
    configs = []
    
    # 1. Standard Grid (Seed)
    grid = np.array([(x, y) for x in range(6) for y in range(5)], dtype=int)
    configs.append(grid)
    
    # 2. Affine Transform of Grid (Shear)
    # Points: (x, y) -> (x + k*y, y)
    for k in [1, 2, -1]:
        pts = []
        for x in range(6):
            for y in range(5):
                pts.append((x + k*y, y))
        # Ensure bounds
        pts = np.array([(px, py) for px, py in pts if -30 <= px <= 30 and -30 <= py <= 30], dtype=int)
        if len(pts) == 30:
            configs.append(pts)
            
    # 3. Parallel Lines (Clustered)
    # Place 6 points on each of 5 parallel lines
    for slope in [0, 1]:
        pts = []
        for line_idx in range(5):
            base_y = line_idx * 2
            for i in range(6):
                if slope == 0:
                    pts.append((i * 2 - 5, base_y))
                else:
                    pts.append((i * 2 - 5, base_y + i * 2 - 5))
        pts = np.array([(px, py) for px, py in pts if -30 <= px <= 30 and -30 <= py <= 30], dtype=int)
        if len(pts) == 30:
            configs.append(pts)
            
    # 4. "Star" or Radial Pattern
    # Center at (0,0), points on concentric circles/rings?
    # Integer points on circles are sparse. 
    # Let's try a "plus" shape expansion.
    pts = []
    # Center
    pts.append((0,0))
    # Rings
    for r in [1, 2, 3, 4, 5, 6]:
        # Add points (r, 0), (0, r), (-r, 0), (0, -r)
        pts.extend([(r, 0), (0, r), (-r, 0), (0, -r)])
    # We have 1 + 4*6 = 25 points. Need 5 more.
    # Add (1,1), (1,-1), (-1,1), (-1,-1), (2,2)
    pts.extend([(1,1), (1,-1), (-1,1), (-1,-1), (2,2)])
    pts = np.array(pts, dtype=int)
    # Check distinctness and bounds
    if len(set(map(tuple, pts))) == 30 and np.all(np.abs(pts) <= 30):
        configs.append(pts)

    # 5. Random but structured: Two grids side by side?
    # Grid 1: x in [0,2], y in [0,9] (3x10) -> 30 points
    pts = np.array([(x, y) for x in range(3) for y in range(10)], dtype=int)
    configs.append(pts)
    
    # Grid 2: x in [0,4], y in [0,5] (5x6) -> 30 points
    pts = np.array([(x, y) for x in range(5) for y in range(6)], dtype=int)
    configs.append(pts)
    
    return configs

def _local_search_area_specific(points, T, time_limit):
    """
    Deterministic local search to maximize count of area T.
    Moves a point if it increases the count of T.
    """
    points = points.copy()
    n = 30
    curr_count = _count_specific_area(points, T)
    best_count = curr_count
    best_points = points.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < time_limit and no_improve_count < max_no_improve:
        improved = False
        # Try moving each point
        for i in range(n):
            if time.time() > time_limit:
                break
            
            old_x, old_y = points[i]
            # Try to move point i to a new location that maximizes T count
            # We can't try all 61x61 locations for every point in every iteration (too slow?)
            # 30 points * 3721 locations * O(N^2) cost per eval?
            # Cost of eval: ~30*30 = 900 ops.
            # Total per full pass: 30 * 3721 * 900 ~ 100M ops. Too slow for Python.
            
            # Instead, use a greedy heuristic for the new location:
            # Pick a random subset of candidate locations or use a grid of candidates.
            # Or, simply try "nudges" (small moves) and "jumps" (random).
            
            # Let's try a mix: 50 random locations + 8 neighbors.
            candidates = set()
            # Neighbors
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = old_x + dx, old_y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.add((nx, ny))
            
            # Random jumps
            for _ in range(20):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                candidates.add((nx, ny))
                
            # Filter out current position and occupied positions
            occupied = set(map(tuple, points))
            occupied.discard((old_x, old_y))
            candidates = [c for c in candidates if c not in occupied]
            
            best_new_loc = None
            best_new_count = curr_count
            
            for nx, ny in candidates:
                points[i] = [nx, ny]
                new_count = _count_specific_area(points, T)
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_new_loc = (nx, ny)
            
            if best_new_loc:
                points[i] = best_new_loc
                curr_count = best_new_count
                improved = True
                if curr_count > best_count:
                    best_count = curr_count
                    best_points = points.copy()
                no_improve_count = 0
            else:
                # Revert if we moved it
                points[i] = [old_x, old_y]
                
        if not improved:
            no_improve_count += 1
            # If stuck, do a random restart or perturb
            # Perturb: move 3 random points to random locations
            for _ in range(3):
                idx = random.randint(0, 29)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                # Ensure distinct
                occupied = set(map(tuple, points))
                occupied.discard(tuple(points[idx]))
                while (nx, ny) in occupied:
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                points[idx] = [nx, ny]
            curr_count = _count_specific_area(points, T)
            if curr_count > best_count:
                best_count = curr_count
                best_points = points.copy()
                
    return best_count, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # 1. Generate initial configs
    configs = _generate_initial_configs()
    
    # 2. Evaluate each to find the most promising target area
    # We will pick the top 3 (score, area) pairs and try to improve them.
    candidates = []
    for cfg in configs:
        if len(set(map(tuple, cfg))) != 30:
            continue
        score, area = _compute_score_fast(cfg)
        if area > 0:
            candidates.append((score, area, cfg))
            
    if not candidates:
        # Fallback to seed
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top unique areas
    seen_areas = set()
    top_candidates = []
    for score, area, cfg in candidates:
        if area not in seen_areas:
            seen_areas.add(area)
            top_candidates.append((score, area, cfg))
        if len(top_candidates) >= 3:
            break
            
    global_best_score = 0
    global_best_points = None
    
    # 3. Run local search for each top candidate
    # Allocate time based on remaining time
    time_left = deadline - time.time()
    time_per_search = time_left / len(top_candidates)
    
    for score, area, cfg in top_candidates:
        if time.time() > deadline - 5:
            break
            
        search_deadline = time.time() + time_per_search
        # Ensure we don't overshoot the global deadline
        search_deadline = min(search_deadline, deadline - 2)
        
        new_score, new_points = _local_search_area_specific(cfg, area, search_deadline)
        
        # Verify final score (might have improved other areas too, but we optimized for 'area')
        # The task asks for the MAX multiplicity of ANY area.
        # Our local search optimized for 'area'. It might have reduced other areas.
        # We should evaluate the full score of the resulting points.
        final_score, final_area = _compute_score_fast(new_points)
        
        if final_score > global_best_score:
            global_best_score = final_score
            global_best_points = new_points
            
    # 4. Fallback
    if global_best_points is None:
        # Return the best initial config
        best_initial = candidates[0][2]
        global_best_points = best_initial
        global_best_score = candidates[0][0]
        
    # 5. Convert to list of tuples
    return [(int(x), int(y)) for x, y in global_best_points]

# EVOLVE-BLOCK-END
