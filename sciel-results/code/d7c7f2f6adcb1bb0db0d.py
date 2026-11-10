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

_DEADLINE = 78.0  # safety margin before 90s kill

def _score_vectorized(points):
    """Compute max multiplicity of any positive doubled area using vectorized numpy."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    # Precompute index arrays for all C(n,3) triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _score_with_areas(points):
    """Return (max_count, area_counts_dict) for detailed analysis."""
    n = len(points)
    if n < 3:
        return 0, {}
    pts = np.array(points, dtype=np.int64)
    
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
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

def _gen_grid_variants():
    """Generate various grid configurations with different spacings."""
    configs = []
    
    # Standard grids with different dimensions and spacings
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3, 4, 5]:
                for dy in [1, 2, 3, 4, 5]:
                    # Center the grid
                    total_w = (cols - 1) * dx
                    total_h = (rows - 1) * dy
                    if total_w > 60 or total_h > 60:
                        continue
                    ox = -total_w // 2
                    oy = -total_h // 2
                    config = _gen_grid_config(rows, cols, dx, dy, (ox, oy))
                    if len(config) == 30:
                        configs.append(config)
    
    # Diagonal grids
    for spacing in [1, 2, 3]:
        for length in range(30, 60):
            pts = set()
            for i in range(length):
                x = -length // 2 + i
                y = -length // 2 + i * spacing
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                configs.append(list(pts))
    
    # Parabolic configurations
    for a in [1, 2, 3, 4]:
        for offset_y in range(-15, 16, 3):
            pts = set()
            for x in range(-15, 16):
                y = a * x * x // 10 + offset_y
                if -30 <= y <= 30:
                    pts.add((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                configs.append(list(pts))
    
    return configs

def _local_search_targeted(points, time_limit=50.0, target_area=None):
    """Local search that targets a specific area value or maximizes max multiplicity."""
    start_time = time.time()
    best_points = list(points)
    best_score, best_areas = _score_with_areas(best_points)
    current_points = list(points)
    current_score, _ = _score_with_areas(current_points)
    
    n = len(current_points)
    
    # If no target area, use the most common area from current config
    if target_area is None:
        _, area_dict = _score_with_areas(current_points)
        if area_dict:
            target_area = max(area_dict, key=area_dict.get)
    
    temp = 3.0
    cooling = 0.997
    min_temp = 0.05
    no_improve_count = 0
    
    while time.time() < time_limit and temp > min_temp:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Generate candidate new positions: nearby points or random
        if np.random.random() < 0.7:
            # Move to a nearby position (within 5 units)
            new_x = old_pt[0] + np.random.randint(-5, 6)
            new_y = old_pt[1] + np.random.randint(-5, 6)
        else:
            new_x = np.random.randint(-30, 31)
            new_y = np.random.randint(-30, 31)
        
        new_pt = (new_x, new_y)
        
        # Check bounds and uniqueness
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            temp *= cooling
            continue
        if new_pt in current_points and new_pt != old_pt:
            temp *= cooling
            continue
        
        # Try the move
        current_points[idx] = new_pt
        new_score, new_areas = _score_with_areas(current_points)
        
        # Check if target area improved
        target_count = new_areas.get(target_area, 0)
        old_target_count = _score_with_areas(current_points)[1].get(target_area, 0) if new_score == current_score else 0
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
        elif target_area and target_count > best_areas.get(target_area, 0):
            # Accept if target area count improved, even if max score didn't
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score or target_count > best_areas.get(target_area, 0):
                best_score = max(best_score, new_score)
                best_points = list(current_points)
                best_areas = new_areas
        else:
            # Accept with probability based on score delta
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.01))
                if np.random.random() > prob:
                    current_points[idx] = old_pt
                else:
                    current_score = new_score
                    no_improve_count += 1
            else:
                current_score = new_score
                no_improve_count += 1
        
        temp *= cooling
        
        # Reset temperature if stuck
        if no_improve_count > 100:
            temp = max(temp, 2.0)
            no_improve_count = 0
    
    return best_points, best_score

def _try_all_configs():
    """Try multiple configurations with targeted local search."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate initial configurations
    configs = _gen_grid_variants()
    
    # Add some random configurations
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            pts.add((x, y))
        configs.append(list(pts))
    
    # Evaluate all configs
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 5.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_with_areas(config)
        scored_configs.append((score, areas, config))
    
    # Sort by score
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2] if scored_configs else _gen_grid_config(5, 6, 1, 1, (-2, -2))
    best_score = scored_configs[0][0] if scored_configs else 0
    best_areas = scored_configs[0][1] if scored_configs else {}
    
    # Run local search on top configurations
    for score, areas, config in scored_configs[:5]:
        if time.time() > deadline - 3.0:
            break
        
        # Use the most common area as target
        if areas:
            target_area = max(areas, key=areas.get)
        else:
            target_area = None
        
        remaining = deadline - time.time() - 2.0
        if remaining > 5.0:
            try:
                lp, ls = _local_search_targeted(config, time_limit=remaining, target_area=target_area)
                if ls > best_score:
                    best_score = ls
                    best_points = lp
                    _, best_areas = _score_with_areas(lp)
            except Exception:
                pass
    
    # If time remains, try more random starts with local search
    if time.time() < deadline - 8.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            try:
                pts = set()
                while len(pts) < 30:
                    x = np.random.randint(-30, 31)
                    y = np.random.randint(-30, 31)
                    pts.add((x, y))
                config = list(pts)
                score, areas = _score_with_areas(config)
                
                if score > best_score:
                    best_score = score
                    best_points = config
                    best_areas = areas
                
                remaining = deadline - time.time() - 3.0
                if remaining > 5.0:
                    target_area = max(areas, key=areas.get) if areas else None
                    lp, ls = _local_search_targeted(config, time_limit=remaining, target_area=target_area)
                    if ls > best_score:
                        best_score = ls
                        best_points = lp
                        _, best_areas = _score_with_areas(lp)
            except Exception:
                pass
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception:
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
