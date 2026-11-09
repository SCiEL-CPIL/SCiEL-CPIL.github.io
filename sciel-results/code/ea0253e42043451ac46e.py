import numpy as np
import time
from math import gcd
from itertools import combinations, product

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence_for_line(A, B, C, points):
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _get_all_lines_and_counts(points):
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
                
    for key in lines_dict:
        A, B, C = key
        lines_dict[key] = _count_incidence_for_line(A, B, C, points)
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    if len(counts) < k:
        return sum(counts.values())
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _evaluate_removal(base_points, num_remove):
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_points = None
    
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        return score, base_points
        
    if num_remove == 1:
        for i in range(n):
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                best_points = base_points[mask]
                
    elif num_remove == 2:
        for i in range(n):
            for j in range(i + 1, n):
                temp_counts = dict(lines_dict)
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = _top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    best_points = base_points[mask]
                    
    else:
        pass
        
    return best_score, best_points

def _generate_grid(a, b, shear=0):
    """Generate a grid with optional shear transformation.
    shear: int, x' = x + shear*y (applied after generating grid)
    """
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    if shear != 0:
        pts[:, 0] = pts[:, 0] + shear * pts[:, 1]
    return pts

def _get_top_lines(points, k=27):
    lines_dict, _ = _get_all_lines_and_counts(points)
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    
    if len(selected) < k:
        all_keys = list(lines_dict.keys())
        sel_keys = set(selected)
        pad_keys = [key for key in all_keys if key not in sel_keys]
        while len(pad_keys) < k - len(selected):
            pad_keys.append((1, 0, 1000000 + len(pad_keys)))
        selected.extend(pad_keys[:k - len(selected)])
        
    return np.array(selected, dtype=np.int64)

def _evaluate_configuration(points):
    lines_dict, _ = _get_all_lines_and_counts(points)
    return _top_k_sum(lines_dict, 27)

def _fast_move_eval(points, move_idx, new_pos):
    """Evaluate the score change from moving point move_idx to new_pos.
    Returns the new top-k sum score, or None if the move is invalid (duplicate point).
    Uses incremental update: remove old point's contributions, add new ones.
    """
    n = len(points)
    old_pos = points[move_idx]
    
    # Check for duplicate
    for i in range(n):
        if i != move_idx and points[i, 0] == new_pos[0] and points[i, 1] == new_pos[1]:
            return None
    
    # Get all lines through old point and all lines through new point
    # and all lines through pairs involving move_idx
    old_x, old_y = int(old_pos[0]), int(old_pos[1])
    new_x, new_y = int(new_pos[0]), int(new_pos[1])
    
    # Compute line counts for all lines, then adjust
    # For efficiency, compute the delta:
    # Lines that lose the old point: lines through (move_idx, j) for all j != move_idx
    # Lines that gain the new point: lines through (move_idx, j) with new position
    # Lines that change: lines through (i, j) where neither is move_idx don't change
    
    # Actually, the simplest correct approach:
    # The set of lines changes:
    # - Lines through old point and point j (j != move_idx): these lines may or may not still exist
    # - Lines through new point and point j (j != move_idx): these lines may or may not exist
    # - Lines not involving move_idx: unchanged
    
    # Let's compute which lines are affected
    affected_lines = {}  # line_key -> net change in count
    
    # Lines that had the old point: for each j != move_idx, line through (old_pos, points[j])
    for j in range(n):
        if j == move_idx:
            continue
        jx, jy = int(points[j, 0]), int(points[j, 1])
        # Line through old_pos and points[j]
        dx = jx - old_x
        dy = jy - old_y
        if dx == 0 and dy == 0:
            continue
        A = dy
        B = -dx
        C = dx * old_y - dy * old_x
        key = _normalize_line(A, B, C)
        if key is not None:
            # This line loses one point (the old position of move_idx)
            # But it might also gain the new point if new_pos is on this line
            if A * new_x + B * new_y + C == 0:
                # Net change: 0 (lost old, gained new)
                pass
            else:
                affected_lines[key] = affected_lines.get(key, 0) - 1
    
    # Lines that will have the new point: for each j != move_idx, line through (new_pos, points[j])
    for j in range(n):
        if j == move_idx:
            continue
        jx, jy = int(points[j, 0]), int(points[j, 1])
        dx = jx - new_x
        dy = jy - new_y
        if dx == 0 and dy == 0:
            continue
        A = dy
        B = -dx
        C = dx * new_y - dy * new_x
        key = _normalize_line(A, B, C)
        if key is not None:
            # Check if this line also passed through old_pos
            if A * old_x + B * old_y + C == 0:
                # Net change: 0 (already counted in the removal above)
                pass
            else:
                # This is a new line or a line that gains the new point
                affected_lines[key] = affected_lines.get(key, 0) + 1
    
    # Now compute the full line counts and apply deltas
    # We need the base counts first
    lines_dict, _ = _get_all_lines_and_counts(points)
    
    # Apply deltas
    for key, delta in affected_lines.items():
        if key in lines_dict:
            lines_dict[key] += delta
        else:
            lines_dict[key] = delta
    
    return _top_k_sum(lines_dict, 27)

def _local_swap_search(points, max_rounds=5, time_budget=None, neighborhood_radius=2):
    """
    Local search: try moving each point to nearby positions.
    Uses fast incremental evaluation.
    """
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    
    if time_budget is not None:
        deadline = time.time() + time_budget
    else:
        deadline = float('inf')
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        n = len(current_points)
        for i in range(n):
            if time.time() > deadline:
                break
            orig_pt = current_points[i]
            ox, oy = int(orig_pt[0]), int(orig_pt[1])
            best_trial_score = current_score
            best_trial_pos = None
            
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (ox + dx, oy + dy)
                    trial_score = _fast_move_eval(current_points, i, cand)
                    if trial_score is not None and trial_score > best_trial_score:
                        best_trial_score = trial_score
                        best_trial_pos = cand
            
            if best_trial_pos is not None:
                current_points[i] = best_trial_pos
                current_score = best_trial_score
                improved = True
        
        if not improved:
            break
    
    return current_score, current_points

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = _get_top_lines(pts, 27)
            
    # 1. Try exact grids (including sheared versions)
    grid_candidates = [
        (6, 9, 0), (9, 6, 0), (3, 18, 0), (18, 3, 0), (2, 27, 0), (27, 2, 0),
        (6, 9, 1), (9, 6, 1), (6, 9, -1), (9, 6, -1),
        (3, 18, 1), (18, 3, 1), (3, 18, -1), (18, 3, -1),
        (4, 14, 0), (14, 4, 0), (4, 14, 1), (14, 4, 1),
        (5, 11, 0), (11, 5, 0), (5, 11, 1), (11, 5, 1),
        (7, 8, 0), (8, 7, 0), (7, 8, 1), (8, 7, 1),
    ]
    for a, b, shear in grid_candidates:
        if time.time() > deadline - 60:
            break
        pts = _generate_grid(a, b, shear)
        if len(pts) != 54:
            continue
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # 2. Try grids with removals
    if time.time() < deadline - 60:
        for a, b, shear in [(7, 8, 0), (8, 7, 0), (5, 11, 0), (11, 5, 0), (4, 14, 0), (14, 4, 0)]:
            if time.time() > deadline - 40:
                break
            pts_base = _generate_grid(a, b, shear)
            num_remove = a * b - 54
            if num_remove < 1 or num_remove > 2:
                continue
            score, pts_best = _evaluate_removal(pts_base, num_remove)
            if pts_best is not None:
                real_score = _evaluate_configuration(pts_best)
                update_best(pts_best, real_score)
            
    # 3. Local search on the best configuration found so far (with wider neighborhood)
    if best_points is not None and time.time() < deadline - 30:
        remaining_time = deadline - 30 - time.time()
        new_score, new_points = _local_swap_search(best_points, max_rounds=5, 
                                                    time_budget=remaining_time * 0.7,
                                                    neighborhood_radius=2)
        if new_score > best_score:
            update_best(new_points, new_score)
    
    # 4. Final polish with even wider neighborhood if time permits
    if best_points is not None and time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        new_score, new_points = _local_swap_search(best_points, max_rounds=3,
                                                    time_budget=remaining_time,
                                                    neighborhood_radius=3)
        if new_score > best_score:
            update_best(new_points, new_score)
        
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    return best_points, best_lines
