"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from collections import Counter

def _evaluate_points(points):
    """
    Evaluate the multiplicity of the most frequent non-zero doubled area
    among all triples of points. Optimized with vectorized numpy operations.
    """
    n = len(points)
    if n < 3:
        return 0
    
    pts = np.array(points, dtype=np.int32)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Vectorized triple enumeration using meshgrid-like approach
    # For n=30, C(30,3) = 4060 triples, so we can afford to build arrays
    i_list = []
    j_list = []
    k_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_list.append(i)
                j_list.append(j)
                k_list.append(k)
    
    if not i_list:
        return 0
        
    i_arr = np.array(i_list, dtype=np.int32)
    j_arr = np.array(j_list, dtype=np.int32)
    k_arr = np.array(k_list, dtype=np.int32)
    
    xi, yi = x[i_arr], y[i_arr]
    xj, yj = x[j_arr], y[j_arr]
    xk, yk = x[k_arr], y[k_arr]
    
    # Determinant: (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    dets = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Exclude zero-area
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
        
    # Use Counter for speed on small arrays
    counts = Counter(dets.tolist())
    return max(counts.values())

def _local_search(points, max_iters=500, time_limit=30.0, neighborhood_radius=5):
    """
    Perform local search by moving points to nearby integer coordinates.
    Uses a larger neighborhood to escape local optima.
    """
    start_time = time.time()
    points = list(points)
    current_score = _evaluate_points(points)
    best_points = list(points)
    best_score = current_score
    
    import random
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, len(points) - 1)
        old_p = points[idx]
        ox, oy = old_p
        
        # Generate a new point in a local neighborhood (larger radius for better exploration)
        dx = rng.randint(-neighborhood_radius, neighborhood_radius)
        dy = rng.randint(-neighborhood_radius, neighborhood_radius)
        new_p = (ox + dx, oy + dy)
        
        # Check bounds
        if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
            continue
            
        # Check if new_p is already in points (excluding the one we're replacing)
        conflict = False
        for i, p in enumerate(points):
            if i != idx and p == new_p:
                conflict = True
                break
        if conflict:
            continue
            
        # Temporarily replace
        points[idx] = new_p
        
        # Evaluate
        new_score = _evaluate_points(points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = list(points)
        else:
            # Revert
            points[idx] = old_p
            
    return best_points, best_score

def _multi_restart_local_search(initial_points, num_restarts=10, time_per_restart=3.0, neighborhood_radius=5):
    """
    Perform multiple restarts of local search from different perturbations.
    """
    start_time = time.time()
    best_points = list(initial_points)
    best_score = _evaluate_points(initial_points)
    
    import random
    rng = random.Random(123)
    
    for r in range(num_restarts):
        if time.time() - start_time > num_restarts * time_per_restart:
            break
        
        # Create a perturbed version
        perturbed = list(initial_points)
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, len(perturbed) - 1)
            ox, oy = perturbed[idx]
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            new_p = (ox + dx, oy + dy)
            if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30:
                conflict = False
                for i, p in enumerate(perturbed):
                    if i != idx and p == new_p:
                        conflict = True
                        break
                if not conflict:
                    perturbed[idx] = new_p
        
        # Run local search from this perturbation
        time_remaining = (num_restarts - r) * time_per_restart - (time.time() - start_time)
        if time_remaining < 0.5:
            break
            
        refined, score = _local_search(perturbed, max_iters=100, time_limit=max(0.5, time_remaining), neighborhood_radius=neighborhood_radius)
        
        if score > best_score:
            best_score = score
            best_points = refined
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Generate various grid-based candidates
    candidates = []
    
    # 6x5 Grid
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(grid_6x5)
    
    # 5x6 Grid
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(grid_5x6)
    
    # 10x3 Grid
    grid_10x3 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(grid_10x3)
    
    # 3x10 Grid
    grid_3x10 = [(x, y) for x in range(3) for y in range(10)]
    candidates.append(grid_3x10)
    
    # 15x2 Grid
    grid_15x2 = [(x, y) for x in range(15) for y in range(2)]
    candidates.append(grid_15x2)
    
    # 2x15 Grid
    grid_2x15 = [(x, y) for x in range(2) for y in range(15)]
    candidates.append(grid_2x15)
    
    # 8x4 Grid minus 2 points
    full_8x4 = [(x, y) for x in range(8) for y in range(4)]
    to_remove_8x4 = [(0, 0), (7, 3)]
    grid_8x4_mod = [p for p in full_8x4 if p not in to_remove_8x4]
    candidates.append(grid_8x4_mod)
    
    # 7x5 Grid minus 5 points
    full_7x5 = [(x, y) for x in range(7) for y in range(5)]
    to_remove_7x5 = [(0, 0), (6, 0), (0, 4), (6, 4), (0, 2)]
    grid_7x5_mod = [p for p in full_7x5 if p not in to_remove_7x5]
    candidates.append(grid_7x5_mod)
    
    # 9x4 Grid minus 6 points
    full_9x4 = [(x, y) for x in range(9) for y in range(4)]
    to_remove_9x4 = [(0, 0), (8, 0), (0, 3), (8, 3), (4, 1), (4, 2)]
    grid_9x4_mod = [p for p in full_9x4 if p not in to_remove_9x4]
    candidates.append(grid_9x4_mod)
    
    # 12x3 Grid minus 6 points
    full_12x3 = [(x, y) for x in range(12) for y in range(3)]
    to_remove_12x3 = [(0, 0), (11, 0), (0, 2), (11, 2), (5, 1), (6, 1)]
    grid_12x3_mod = [p for p in full_12x3 if p not in to_remove_12x3]
    candidates.append(grid_12x3_mod)
    
    best_points = grid_6x5
    best_score = 0
    
    # Evaluate all candidates
    for pts in candidates:
        if time.time() > deadline - 20.0:
            break
        if len(pts) != 30:
            continue
        valid = all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts)
        if not valid:
            continue
        score = _evaluate_points(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            
    # Phase 1: Local search with larger neighborhood
    if time.time() < deadline - 10.0:
        time_remaining = deadline - time.time() - 5.0
        refined_points, refined_score = _local_search(
            best_points, 
            max_iters=300, 
            time_limit=time_remaining,
            neighborhood_radius=5
        )
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
            
    # Phase 2: Multi-restart local search
    if time.time() < deadline - 2.0:
        time_remaining = deadline - time.time() - 1.0
        num_restarts = min(10, int(time_remaining / 2.0))
        if num_restarts > 0:
            multi_points, multi_score = _multi_restart_local_search(
                best_points,
                num_restarts=num_restarts,
                time_per_restart=time_remaining / max(num_restarts, 1),
                neighborhood_radius=5
            )
            if multi_score > best_score:
                best_points = multi_points
                best_score = multi_score
                
    return best_points
# EVOLVE-BLOCK-END
