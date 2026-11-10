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

_DEADLINE = 82.0  # safety margin before 90s kill

# Precompute triple indices for n=30 once
_N = 30
_i_idx, _j_idx, _k_idx = [], [], []
for _i in range(_N - 2):
    for _j in range(_i + 1, _N - 1):
        for _k in range(_j + 1, _N):
            _i_idx.append(_i)
            _j_idx.append(_j)
            _k_idx.append(_k)
_I_IDX = np.array(_i_idx, dtype=np.int32)
_J_IDX = np.array(_j_idx, dtype=np.int32)
_K_IDX = np.array(_k_idx, dtype=np.int32)
_NUM_TRIPLES = len(_I_IDX)  # C(30,3) = 4060

def _score_fast(points):
    """Fast scoring using precomputed indices. Returns (max_count, area_counts_dict)."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
    vals, counts = np.unique(det, return_counts=True)
    area_dict = dict(zip(vals.tolist(), counts.tolist()))
    return int(counts.max()), area_dict

def _score_target_area(points, target_area):
    """Count how many triangles have exactly target_area (doubled)."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return int(np.sum(det == target_area))

def _gen_grid_config(rows, cols, dx, dy, x_offset=0, y_offset=0):
    """Generate a rectangular grid of points."""
    pts = []
    total_w = (cols - 1) * dx
    total_h = (rows - 1) * dy
    ox = -total_w // 2 + x_offset
    oy = -total_h // 2 + y_offset
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts

def _gen_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Rectangular grids with various spacings
    for rows in range(3, 8):
        for cols in range(3, 11):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    pts = _gen_grid_config(rows, cols, dx, dy)
                    if len(pts) >= 30:
                        configs.append(sorted(pts)[:30])
    
    # Hexagonal lattice
    for scale in [2, 3, 4]:
        pts = set()
        for q in range(-5, 6):
            for r in range(-5, 6):
                s = -q - r
                if abs(s) > 5:
                    continue
                x = (q + s // 2) * scale
                y = (r + (q % 2) * 0) * scale
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
        if len(pts) >= 30:
            configs.append(sorted(pts)[:30])
    
    # Lattice-based configurations
    for a, b in [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]:
        for c, d in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]:
            if a * d - b * c == 0:
                continue
            pts = set()
            for i in range(-8, 9):
                for j in range(-8, 9):
                    x = i * a + j * c
                    y = i * b + j * d
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.add((x, y))
            if len(pts) >= 30:
                configs.append(sorted(pts)[:30])
    
    # Random configurations with seeded RNG
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-25, 26)
            y = rng.randint(-25, 26)
            pts.add((x, y))
        configs.append(sorted(pts))
    
    # Remove duplicates
    unique_configs = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    
    return unique_configs

def _targeted_area_search(points, deadline, candidate_areas=None):
    """
    Structural change: Instead of random SA, use targeted area-multiplicity optimization.
    For each candidate area value, greedily try to increase the count of triangles
    with that area by moving points to positions that create more such triangles.
    """
    start_time = time.time()
    current_points = list(points)
    n = len(current_points)
    point_set = set(current_points)
    
    # Get current area distribution
    _, current_areas = _score_fast(current_points)
    
    # Candidate areas: focus on small values that tend to have high multiplicity
    if candidate_areas is None:
        # Prioritize small areas (1-20) and any areas already present with count > 5
        candidate_areas = list(range(1, 21))
        for area, count in current_areas.items():
            if count > 5 and area not in candidate_areas:
                candidate_areas.append(area)
        candidate_areas.sort()
    
    best_points = list(current_points)
    best_score = max(current_areas.values()) if current_areas else 0
    best_area = max(current_areas, key=current_areas.get) if current_areas else None
    
    # For each candidate area, try to maximize its multiplicity
    for target_area in candidate_areas:
        if time.time() > deadline:
            break
        
        # Work on a copy
        work_points = list(current_points)
        work_set = set(work_points)
        
        # Try multiple rounds of improvement for this target area
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                old_pt = work_points[idx]
                best_move = None
                best_count = _score_target_area(work_points, target_area)
                
                # Try nearby positions
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                            continue
                        if new_pt in work_set:
                            continue
                        
                        # Try this move
                        work_set.discard(old_pt)
                        work_set.add(new_pt)
                        work_points[idx] = new_pt
                        
                        new_count = _score_target_area(work_points, target_area)
                        
                        if new_count > best_count:
                            best_count = new_count
                            best_move = new_pt
                        else:
                            # Revert
                            work_points[idx] = old_pt
                            work_set.discard(new_pt)
                            work_set.add(old_pt)
                
                if best_move is not None:
                    # Apply the best move
                    work_set.discard(old_pt)
                    work_set.add(best_move)
                    work_points[idx] = best_move
                    improved = True
        
        # Check if this improved the overall score
        work_score, work_areas = _score_fast(work_points)
        if work_score > best_score:
            best_score = work_score
            best_points = work_points
            best_area = max(work_areas, key=work_areas.get) if work_areas else None
            current_points = work_points
            current_areas = work_areas
    
    return best_points, best_score

def _hybrid_search(points, deadline):
    """
    Combine targeted area search with a final polish pass.
    """
    start_time = time.time()
    
    # Phase 1: Targeted area search (structural change)
    if time.time() < deadline - 15.0:
        p1, s1 = _targeted_area_search(points, time.time() + (deadline - start_time - 15.0))
        if s1 >= _score_fast(points)[0]:
            points = p1
    
    # Phase 2: Final polish - exhaustive single-point moves
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    point_set = set(current_points)
    n = len(current_points)
    
    if current_areas:
        target_area = max(current_areas, key=current_areas.get)
    else:
        target_area = None
    
    # Multiple polish passes with shrinking neighborhood
    for radius in [3, 2, 1]:
        if time.time() > deadline:
            break
        for idx in range(n):
            if time.time() > deadline:
                break
            old_pt = current_points[idx]
            best_for_idx = None
            best_for_idx_score = current_score
            best_for_idx_target = current_areas.get(target_area, 0) if target_area else 0
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                        continue
                    if new_pt in point_set:
                        continue
                    
                    point_set.discard(old_pt)
                    point_set.add(new_pt)
                    current_points[idx] = new_pt
                    
                    sc, ar = _score_fast(current_points)
                    tc = ar.get(target_area, 0) if target_area else 0
                    
                    if sc > best_for_idx_score or (sc == best_for_idx_score and tc > best_for_idx_target):
                        best_for_idx_score = sc
                        best_for_idx_target = tc
                        best_for_idx = new_pt
                    else:
                        current_points[idx] = old_pt
                        point_set.discard(new_pt)
                        point_set.add(old_pt)
            
            if best_for_idx is not None:
                point_set.discard(old_pt)
                point_set.add(best_for_idx)
                current_points[idx] = best_for_idx
                current_score, current_areas = _score_fast(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
    
    return best_points, best_score

def _try_all_configs():
    """Main search: evaluate initial configs, then hybrid search on top ones."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 20.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    
    # Run hybrid search on top configurations
    top_k = min(6, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 5.0:
            break
        score, areas, config = scored_configs[i]
        remaining = deadline - time.time() - 3.0
        if remaining < 5.0:
            break
        try:
            lp, ls = _hybrid_search(config, time.time() + remaining)
            if ls > best_score:
                best_score = ls
                best_points = lp
        except Exception:
            pass
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    
    # Verify validity
    pts_set = set(result)
    if len(pts_set) != 30:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
            break
    
    return result
# EVOLVE-BLOCK-END
