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

_DEADLINE = 50.0  # safety margin before 90s kill

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

def _score_max_only(points):
    """Return just the max multiplicity."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _gen_rect_grid(rows, cols, dx, dy, ox=0, oy=0):
    """Generate a centered rectangular grid, return exactly 30 points or None."""
    pts = set()
    total_w = (cols - 1) * dx
    total_h = (rows - 1) * dy
    if total_w > 60 or total_h > 60:
        return None
    ox = -total_w // 2 + ox
    oy = -total_h // 2 + oy
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    if len(pts) < 30:
        return None
    return sorted(pts)[:30]

def _gen_hex_config(radius, scale=2):
    """Generate a hexagonal lattice configuration."""
    pts = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if abs(s) > radius:
                continue
            x = q + s // 2
            y = r
            cx = x * scale
            cy = y * scale + (q % 2) * scale // 2
            if -30 <= cx <= 30 and -30 <= cy <= 30:
                pts.add((cx, cy))
    if len(pts) < 30:
        return None
    return sorted(pts)[:30]

def _gen_lattice(a, b, c, d, count=30):
    """Generate points on a 2D lattice spanned by vectors (a,b) and (c,d)."""
    pts = set()
    for i in range(-15, 16):
        for j in range(-15, 16):
            x = i * a + j * c
            y = i * b + j * d
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    if len(pts) < count:
        return None
    return sorted(pts)[:count]

def _gen_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Rectangular grids with various offsets
    for rows in range(3, 9):
        for cols in range(3, 11):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    for ox in [-2, -1, 0, 1, 2]:
                        for oy in [-2, -1, 0, 1, 2]:
                            cfg = _gen_rect_grid(rows, cols, dx, dy, ox, oy)
                            if cfg and len(cfg) == 30:
                                configs.append(cfg)
    
    # Hexagonal
    for r in range(3, 7):
        for scale in [1, 2, 3]:
            cfg = _gen_hex_config(r, scale)
            if cfg:
                configs.append(cfg)
    
    # Various lattices
    lattice_pairs = [
        (1, 0, 0, 1), (2, 0, 0, 1), (1, 1, 0, 1), (2, 1, 0, 1),
        (1, 2, 0, 1), (3, 1, 0, 1), (1, 0, 1, 1), (2, 0, 1, 1),
        (1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 1, 1), (3, 1, 1, 1),
        (1, 0, 1, 2), (2, 0, 1, 2), (1, 1, 1, 2), (2, 1, 1, 2),
        (1, 0, 1, 3), (2, 0, 1, 3), (1, 1, 1, 3), (2, 1, 1, 3),
        (1, 0, 2, 1), (2, 0, 2, 1), (1, 1, 2, 1), (2, 1, 2, 1),
        (1, 0, 2, 2), (2, 0, 2, 2), (1, 1, 2, 2), (2, 1, 2, 2),
        (1, 0, 3, 1), (2, 0, 3, 1), (1, 1, 3, 1), (2, 1, 3, 1),
        (1, 0, 3, 2), (2, 0, 3, 2), (1, 1, 3, 2), (2, 1, 3, 2),
        (1, 0, 3, 3), (2, 0, 3, 3), (1, 1, 3, 3), (2, 1, 3, 3),
        (1, 1, 2, 3), (1, 2, 2, 3), (1, 3, 2, 3), (2, 1, 3, 2),
        (2, 2, 3, 1), (2, 3, 3, 1), (3, 1, 3, 2), (3, 2, 3, 1),
    ]
    for a, b, c, d in lattice_pairs:
        if a * d - b * c == 0:
            continue
        cfg = _gen_lattice(a, b, c, d)
        if cfg:
            configs.append(cfg)
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        rng = np.random.RandomState(seed)
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-30, 31)
            y = rng.randint(-30, 31)
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

def _local_search(points, deadline, target_area=None, rng_seed=42):
    """Simulated annealing local search with target area focus."""
    start_time = time.time()
    rng = np.random.RandomState(rng_seed)
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    best_areas = dict(current_areas)
    
    n = len(current_points)
    point_set = set(current_points)
    
    if target_area is None and current_areas:
        target_area = max(current_areas, key=current_areas.get)
    
    temp = 5.0
    cooling = 0.995
    min_temp = 0.01
    no_improve = 0
    steps = 0
    
    while time.time() < deadline and temp > min_temp:
        # Pick a point to move
        idx = rng.randint(n)
        old_pt = current_points[idx]
        
        # Generate new position - bias toward small moves for refinement
        r = rng.random()
        if r < 0.5:
            # Small perturbation
            radius = rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
            new_x = old_pt[0] + rng.randint(-radius, radius + 1)
            new_y = old_pt[1] + rng.randint(-radius, radius + 1)
        elif r < 0.8:
            # Medium perturbation
            new_x = old_pt[0] + rng.randint(-8, 9)
            new_y = old_pt[1] + rng.randint(-8, 9)
        else:
            # Large perturbation
            new_x = rng.randint(-30, 31)
            new_y = rng.randint(-30, 31)
        
        new_pt = (new_x, new_y)
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            temp *= cooling
            continue
        if new_pt != old_pt and new_pt in point_set:
            temp *= cooling
            continue
        
        # Apply move
        point_set.discard(old_pt)
        point_set.add(new_pt)
        current_points[idx] = new_pt
        
        new_score, new_areas = _score_fast(current_points)
        steps += 1
        
        # Acceptance criterion - prioritize improving target area count
        target_improved = False
        if target_area is not None:
            old_target_count = current_areas.get(target_area, 0)
            new_target_count = new_areas.get(target_area, 0)
            target_improved = new_target_count > old_target_count
        
        if new_score > current_score:
            current_score = new_score
            current_areas = new_areas
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
        elif target_improved:
            # Accept if target area count improves (even if overall score same)
            current_score = new_score
            current_areas = new_areas
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.01))
                if rng.random() > prob:
                    # Revert
                    current_points[idx] = old_pt
                    point_set.discard(new_pt)
                    point_set.add(old_pt)
                    no_improve += 1
                else:
                    current_score = new_score
                    current_areas = new_areas
                    no_improve += 1
            else:
                current_score = new_score
                current_areas = new_areas
                no_improve += 1
        
        temp *= cooling
        
        # Reheat if stuck
        if no_improve > 150:
            temp = max(temp, 3.0)
            no_improve = 0
    
    return best_points, best_score

def _polish(points, deadline):
    """Final high-precision polish: try all single-point moves to neighboring cells."""
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
    
    # Multiple passes with decreasing neighborhood size
    for radius in [4, 3, 2, 1]:
        for idx in range(n):
            if time.time() > deadline:
                return best_points, best_score
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
                    if new_pt != old_pt and new_pt in point_set:
                        continue
                    
                    # Try this move
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
                        # Revert
                        current_points[idx] = old_pt
                        point_set.discard(new_pt)
                        point_set.add(old_pt)
            
            if best_for_idx is not None:
                # Apply the best move for this index
                point_set.discard(old_pt)
                point_set.add(best_for_idx)
                current_points[idx] = best_for_idx
                current_score, current_areas = _score_fast(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
    
    return best_points, best_score

def _greedy_set_cover(points, target_area, deadline):
    """
    Greedy set-cover approach: try to maximize the count of triangles with target_area.
    For each candidate point position, compute how many new triangles of target_area it creates.
    """
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    best_target_count = current_areas.get(target_area, 0)
    
    n = len(current_points)
    point_set = set(current_points)
    
    # Try to improve by moving points to positions that create more target_area triangles
    for iteration in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                return best_points, best_score
            
            old_pt = current_points[idx]
            best_move = None
            best_move_score = best_score
            best_move_target = best_target_count
            
            # Try nearby positions
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                        continue
                    if new_pt != old_pt and new_pt in point_set:
                        continue
                    
                    # Try this move
                    point_set.discard(old_pt)
                    point_set.add(new_pt)
                    current_points[idx] = new_pt
                    
                    sc, ar = _score_fast(current_points)
                    tc = ar.get(target_area, 0)
                    
                    if sc > best_move_score or (sc == best_move_score and tc > best_move_target):
                        best_move_score = sc
                        best_move_target = tc
                        best_move = new_pt
                    else:
                        # Revert
                        current_points[idx] = old_pt
                        point_set.discard(new_pt)
                        point_set.add(old_pt)
            
            if best_move is not None:
                # Apply the best move for this index
                point_set.discard(old_pt)
                point_set.add(best_move)
                current_points[idx] = best_move
                current_score, current_areas = _score_fast(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
                    best_target_count = current_areas.get(target_area, 0)
                improved = True
        
        if not improved:
            break
    
    return best_points, best_score

def _try_all_configs():
    """Main search: evaluate initial configs, then local search on top ones."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        # Fallback
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    best_areas = scored_configs[0][1]
    
    # Run local search on top configurations with more time allocation
    top_k = min(8, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 8.0:
            break
        score, areas, config = scored_configs[i]
        target_area = max(areas, key=areas.get) if areas else None
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        try:
            lp, ls = _local_search(config, time.time() + remaining, target_area=target_area, rng_seed=42 + i)
            if ls > best_score:
                best_score = ls
                best_points = lp
                _, best_areas = _score_fast(lp)
        except Exception:
            pass
    
    # Polish the best result
    if time.time() < deadline - 3.0:
        try:
            pp, ps = _polish(best_points, deadline - 1.0)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    # Additional greedy set-cover refinement on the best result
    if time.time() < deadline - 2.0:
        try:
            target_area = max(best_areas, key=best_areas.get) if best_areas else None
            if target_area:
                gp, gs = _greedy_set_cover(best_points, target_area, deadline - 0.5)
                if gs > best_score:
                    best_score = gs
                    best_points = gp
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
