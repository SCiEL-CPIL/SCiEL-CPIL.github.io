"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import Counter

_DEADLINE = 75.0  # safety margin before 90s kill
_START_TIME = time.time()

# Precompute triple indices for n=30
_N = 30
_TRIPLE_INDICES = None

def _get_triple_indices():
    global _TRIPLE_INDICES
    if _TRIPLE_INDICES is None:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(_N - 2):
            for j in range(i + 1, _N - 1):
                for k in range(j + 1, _N):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _TRIPLE_INDICES = (np.array(i_idx, dtype=np.int32), np.array(j_idx, dtype=np.int32), np.array(k_idx, dtype=np.int32))
    return _TRIPLE_INDICES

def _score_fast(points):
    """
    Fast scoring function using precomputed indices.
    Returns the maximum multiplicity of any positive doubled area.
    """
    if len(points) != _N:
        return 0
    
    pts = np.array(points, dtype=np.int32)
    i_idx, j_idx, k_idx = _get_triple_indices()
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    # Calculate doubled area (absolute determinant)
    # det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Exclude zero area
    det = det[det > 0]
    if len(det) == 0:
        return 0
        
    # Count occurrences
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _score_with_areas_fast(points):
    """
    Returns (max_count, area_counts_dict) using precomputed indices.
    """
    if len(points) != _N:
        return 0, {}
        
    pts = np.array(points, dtype=np.int32)
    i_idx, j_idx, k_idx = _get_triple_indices()
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
        
    vals, counts = np.unique(det, return_counts=True)
    area_dict = dict(zip(vals.tolist(), counts.tolist()))
    return int(counts.max()), area_dict

def _gen_grid_config(rows, cols, dx=1, dy=1, offset=(0, 0)):
    """Generate a grid configuration with given parameters."""
    pts = set()
    ox, oy = offset
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    return list(pts)[:30]

def _gen_initial_configs():
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # 1. Standard grids
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    total_w = (cols - 1) * dx
                    total_h = (rows - 1) * dy
                    if total_w > 60 or total_h > 60:
                        continue
                    ox = -total_w // 2
                    oy = -total_h // 2
                    config = _gen_grid_config(rows, cols, dx, dy, (ox, oy))
                    if len(config) == 30:
                        configs.append(config)

    # 2. Random configurations
    for _ in range(5):
        pts = set()
        while len(pts) < 30:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            pts.add((x, y))
        configs.append(list(pts))
        
    return configs

def _greedy_construct(target_area):
    """
    Greedily select 30 points to maximize the count of triangles with area == target_area.
    This is a heuristic to find a good starting point for local search.
    """
    # Start with a random point
    points = []
    used = set()
    
    # Pick first 3 points randomly to define initial triangles
    for _ in range(3):
        while True:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            p = (x, y)
            if p not in used:
                points.append(p)
                used.add(p)
                break
                
    # For the next 27 points, pick the one that maximizes the number of new triangles with target_area
    # formed with the existing points.
    for _ in range(27):
        best_pt = None
        best_count = -1
        # Sample a subset of candidate points to keep this fast
        # Instead of checking all 3600 points, check a random sample or a grid
        candidates = set()
        # Add random candidates
        for _ in range(100):
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            candidates.add((x, y))
        # Add points near existing points
        for p in points:
            for dx in range(-5, 6, 2):
                for dy in range(-5, 6, 2):
                    nx, ny = p[0] + dx, p[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.add((nx, ny))
                        
        for cand in candidates:
            if cand in used:
                continue
            
            # Calculate how many triangles with area target_area this candidate forms with existing pairs
            count = 0
            n_existing = len(points)
            for i in range(n_existing):
                for j in range(i + 1, n_existing):
                    p1 = points[i]
                    p2 = points[j]
                    # Determinant of (p2-p1) and (cand-p1)
                    det = abs((p2[0] - p1[0]) * (cand[1] - p1[1]) - (p2[1] - p1[1]) * (cand[0] - p1[0]))
                    if det == target_area:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_pt = cand
                
        if best_pt is not None:
            points.append(best_pt)
            used.add(best_pt)
            
    return points

def _local_search(points, target_area, time_limit=5.0):
    """
    Local search to maximize the count of triangles with area == target_area.
    """
    start_time = time.time()
    best_points = list(points)
    best_count = _count_area(best_points, target_area)
    current_points = list(points)
    current_count = best_count
    
    n = len(current_points)
    temp = 1.0
    cooling = 0.99
    
    while time.time() < start_time + time_limit:
        if time.time() > _START_TIME + _DEADLINE:
            break
            
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Generate new position
        if np.random.random() < 0.8:
            # Local move
            new_x = old_pt[0] + np.random.randint(-3, 4)
            new_y = old_pt[1] + np.random.randint(-3, 4)
        else:
            # Random jump
            new_x = np.random.randint(-30, 31)
            new_y = np.random.randint(-30, 31)
            
        new_pt = (new_x, new_y)
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        if new_pt in current_points and new_pt != old_pt:
            continue
            
        # Calculate new count efficiently? 
        # For now, we use the full scorer but filtered to target area if possible, 
        # but _count_area is faster than _score_fast if we only care about one area.
        
        current_points[idx] = new_pt
        new_count = _count_area(current_points, target_area)
        
        if new_count > current_count:
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_points = list(current_points)
        else:
            # Accept with probability
            delta = new_count - current_count
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.1))
                if np.random.random() > prob:
                    current_points[idx] = old_pt
            else:
                current_count = new_count
                
        temp *= cooling
        
    return best_points, best_count

def _count_area(points, target_area):
    """Count triangles with exactly target_area."""
    pts = np.array(points, dtype=np.int32)
    i_idx, j_idx, k_idx = _get_triple_indices()
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Count matches
    return int(np.sum(det == target_area))

def _try_all_configs():
    """Main search logic."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    configs = _gen_initial_configs()
    
    # Evaluate initial configs to find good target areas
    best_score = 0
    best_points = configs[0]
    target_areas = set()
    
    for config in configs:
        if time.time() > deadline - 10.0:
            break
        score, areas = _score_with_areas_fast(config)
        if score > best_score:
            best_score = score
            best_points = config
        # Add top areas to target set
        if areas:
            sorted_areas = sorted(areas.items(), key=lambda x: -x[1])
            for area, count in sorted_areas[:3]:
                if count > 10: # Only target areas that appear frequently
                    target_areas.add(area)
    
    # If no good target areas found, use some small integers
    if not target_areas:
        target_areas = {1, 2, 3, 4, 5}
        
    # Run greedy + local search for each target area
    for target_area in target_areas:
        if time.time() > deadline - 15.0:
            break
            
        # Generate a few greedy starts for this target area
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
                
            greedy_pts = _greedy_construct(target_area)
            
            # Local search to refine
            remaining_time = deadline - time.time() - 5.0
            if remaining_time < 2.0:
                break
                
            ls_pts, ls_count = _local_search(greedy_pts, target_area, time_limit=remaining_time * 0.8)
            
            # Evaluate the final result with the full scorer (max multiplicity)
            final_score, final_areas = _score_with_areas_fast(ls_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = ls_pts
                
            # Also check if the greedy start itself was better
            greedy_score, _ = _score_with_areas_fast(greedy_pts)
            if greedy_score > best_score:
                best_score = greedy_score
                best_points = greedy_pts
                
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception as e:
        # Fallback to simple grid
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
        
    # Verify validity
    pts = set(result)
    if len(pts) != 30:
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
            break
            
    return result
# EVOLVE-BLOCK-END
