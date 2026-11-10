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

def _compute_all_dets(pts):
    """Compute all triple determinants for a point array."""
    i_idx, j_idx, k_idx = _get_triple_indices()
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return det

def _incremental_score(pts, idx_new, new_pt):
    """
    Compute the score change when replacing point idx_new with new_pt.
    Returns (new_max_count, new_area_counts) without full recomputation.
    """
    n = len(pts)
    # Remove contributions involving old point idx_new
    # Add contributions involving new point new_pt
    
    # For speed, just do full recompute but with numpy - it's fast enough for n=30
    pts_copy = pts.copy()
    pts_copy[idx_new] = new_pt
    det = _compute_all_dets(pts_copy)
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max()), dict(zip(vals.tolist(), counts.tolist()))

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

def _gen_all_grids():
    """Systematically generate all valid rectangular grid configurations."""
    configs = []
    seen = set()
    
    for rows in range(1, 31):
        for cols in range(1, 31):
            if rows * cols < 30:
                continue
            if rows * cols > 30:
                # Can't fit exactly 30 in a full rectangle unless we trim
                # Skip for now; handle separately
                continue
            for dx in range(1, 8):
                for dy in range(1, 8):
                    total_w = (cols - 1) * dx
                    total_h = (rows - 1) * dy
                    if total_w > 60 or total_h > 60:
                        continue
                    # Try different offsets to center
                    for ox in range(-total_w // 2, total_w // 2 + 1, max(1, dx)):
                        for oy in range(-total_h // 2, total_h // 2 + 1, max(1, dy)):
                            config = _gen_grid_config(rows, cols, dx, dy, (ox, oy))
                            if len(config) == 30:
                                key = frozenset(config)
                                if key not in seen:
                                    seen.add(key)
                                    configs.append(config)
    
    # Also try grids that are subsets of larger grids (e.g., 6x5=30 exactly)
    # 30 = 2*15, 3*10, 5*6, 6*5, 10*3, 15*2
    for rows, cols in [(2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2)]:
        for dx in range(1, 8):
            for dy in range(1, 8):
                total_w = (cols - 1) * dx
                total_h = (rows - 1) * dy
                if total_w > 60 or total_h > 60:
                    continue
                ox = -total_w // 2
                oy = -total_h // 2
                config = _gen_grid_config(rows, cols, dx, dy, (ox, oy))
                if len(config) == 30:
                    key = frozenset(config)
                    if key not in seen:
                        seen.add(key)
                        configs.append(config)
    
    return configs

def _gen_initial_configs():
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # 1. All valid rectangular grids
    configs.extend(_gen_all_grids())
    
    # 2. Random configurations
    for _ in range(10):
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
    points = []
    used = set()
    
    # Pick first 3 points randomly
    for _ in range(3):
        while True:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            p = (x, y)
            if p not in used:
                points.append(p)
                used.add(p)
                break
                
    # For the next 27 points, pick the one that maximizes new target-area triangles
    for _ in range(27):
        best_pt = None
        best_count = -1
        
        # Build candidate set
        candidates = set()
        for _ in range(200):
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            candidates.add((x, y))
        for p in points:
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    nx, ny = p[0] + dx, p[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.add((nx, ny))
                        
        for cand in candidates:
            if cand in used:
                continue
            
            count = 0
            n_existing = len(points)
            for i in range(n_existing):
                for j in range(i + 1, n_existing):
                    p1 = points[i]
                    p2 = points[j]
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
    Uses vectorized scoring for speed.
    """
    start_time = time.time()
    best_points = list(points)
    best_count = _count_area(best_points, target_area)
    current_points = list(points)
    current_count = best_count
    
    n = len(current_points)
    temp = 1.0
    cooling = 0.995
    
    while time.time() < start_time + time_limit:
        if time.time() > _START_TIME + _DEADLINE:
            break
            
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Generate new position
        if np.random.random() < 0.7:
            # Local move
            new_x = old_pt[0] + np.random.randint(-4, 5)
            new_y = old_pt[1] + np.random.randint(-4, 5)
        elif np.random.random() < 0.5:
            # Medium jump
            new_x = old_pt[0] + np.random.randint(-15, 16)
            new_y = old_pt[1] + np.random.randint(-15, 16)
        else:
            # Random jump
            new_x = np.random.randint(-30, 31)
            new_y = np.random.randint(-30, 31)
            
        new_pt = (new_x, new_y)
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        if new_pt in current_points and new_pt != old_pt:
            continue
            
        current_points[idx] = new_pt
        new_count = _count_area(current_points, target_area)
        
        if new_count > current_count:
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_points = list(current_points)
        else:
            delta = new_count - current_count
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.05))
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
    best_points = configs[0] if configs else [(i, 0) for i in range(30)]
    target_areas = set()
    
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        score, areas = _score_with_areas_fast(config)
        if score > best_score:
            best_score = score
            best_points = config
        # Add top areas to target set
        if areas:
            sorted_areas = sorted(areas.items(), key=lambda x: -x[1])
            for area, count in sorted_areas[:5]:
                if count > 5:
                    target_areas.add(area)
    
    # If no good target areas found, use small integers
    if not target_areas:
        target_areas = {1, 2, 3, 4, 5, 6, 8}
    
    # Run local search from best grid configs and greedy starts
    # First: local search from top grid configs
    top_configs = []
    for config in configs[:20]:  # Limit to top 20 configs
        score, _ = _score_with_areas_fast(config)
        top_configs.append((score, config))
    top_configs.sort(key=lambda x: -x[0])
    
    for score, config in top_configs[:8]:
        if time.time() > deadline - 10.0:
            break
        # Local search on the best area for this config
        _, areas = _score_with_areas_fast(config)
        if areas:
            top_area = max(areas, key=areas.get)
            remaining = deadline - time.time() - 3.0
            if remaining > 3.0:
                ls_pts, _ = _local_search(config, top_area, time_limit=min(remaining * 0.9, 8.0))
                final_score, _ = _score_with_areas_fast(ls_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_points = ls_pts
    
    # Then: greedy + local search for each target area
    for target_area in target_areas:
        if time.time() > deadline - 8.0:
            break
            
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
                
            greedy_pts = _greedy_construct(target_area)
            
            remaining_time = deadline - time.time() - 3.0
            if remaining_time < 2.0:
                break
                
            ls_pts, _ = _local_search(greedy_pts, target_area, time_limit=min(remaining_time * 0.85, 6.0))
            
            final_score, _ = _score_with_areas_fast(ls_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = ls_pts
            
            greedy_score, _ = _score_with_areas_fast(greedy_pts)
            if greedy_score > best_score:
                best_score = greedy_score
                best_points = greedy_pts
                
    # Final polish: try to improve best_points with multi-target local search
    if time.time() < deadline - 3.0:
        _, final_areas = _score_with_areas_fast(best_points)
        if final_areas:
            top_area = max(final_areas, key=final_areas.get)
            remaining = deadline - time.time() - 1.0
            if remaining > 2.0:
                polished, _ = _local_search(best_points, top_area, time_limit=remaining * 0.9)
                polished_score, _ = _score_with_areas_fast(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception as e:
        # Fallback to 6x5 grid centered at origin
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
