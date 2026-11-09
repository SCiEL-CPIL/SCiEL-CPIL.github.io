# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation Ax + By + C = 0 to canonical form."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _get_all_lines_and_counts(points):
    """
    Generate all lines through pairs of points and count incidences.
    Returns:
        lines_dict: { (A,B,C): count }
        point_lines: list of sets of line keys for each point
    """
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = int(p2[0]) - int(p1[0])
            dy = int(p2[1]) - int(p1[1])
            A = dy
            B = -dx
            C = dx * int(p1[1]) - dy * int(p1[0])
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
                
    if lines_dict:
        keys = list(lines_dict.keys())
        A_arr = np.array([k[0] for k in keys], dtype=np.int64)
        B_arr = np.array([k[1] for k in keys], dtype=np.int64)
        C_arr = np.array([k[2] for k in keys], dtype=np.int64)
        xs = points[:, 0].astype(np.int64)
        ys = points[:, 1].astype(np.int64)
        vals = A_arr[:, None] * xs[None, :] + B_arr[:, None] * ys[None, :] + C_arr[:, None]
        counts = np.sum(vals == 0, axis=1)
        for idx, key in enumerate(keys):
            lines_dict[key] = int(counts[idx])
        
    return lines_dict, point_lines

def _top_k_sum_from_array(arr, k=27):
    """Return sum of top k values in a numpy array."""
    if len(arr) < k:
        return int(np.sum(arr))
    top_k = np.partition(arr, -k)[-k:]
    return int(np.sum(top_k))

def _select_top_lines(lines_dict, k=27):
    """Select top k lines by incidence count."""
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    if len(selected) < k:
        sel_keys = set(selected)
        for key in lines_dict:
            if len(selected) >= k:
                break
            if key not in sel_keys:
                selected.append(key)
        while len(selected) < k:
            selected.append((1, 0, 1000000 + len(selected)))
    return np.array(selected, dtype=np.int64)

def _compute_incidence(pts, lines):
    """Compute exact incidence count."""
    A_arr = lines[:, 0].astype(np.int64)
    B_arr = lines[:, 1].astype(np.int64)
    C_arr = lines[:, 2].astype(np.int64)
    xs = pts[:, 0].astype(np.int64)
    ys = pts[:, 1].astype(np.int64)
    vals = A_arr[:, None] * xs[None, :] + B_arr[:, None] * ys[None, :] + C_arr[:, None]
    return int(np.sum(vals == 0))

def _generate_grid(a, b):
    """Generate an a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _try_exact_grid(a, b):
    """Try an exact a*b = 54 grid."""
    if a * b != 54:
        return None, None
    pts = _generate_grid(a, b)
    lines_dict, _ = _get_all_lines_and_counts(pts)
    lines_arr = _select_top_lines(lines_dict)
    return pts, lines_arr

def _evaluate_removal(base_points, num_remove, deadline):
    """
    Evaluate removing num_remove points from base_points to maximize top-27 line incidences.
    """
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_mask = None
    
    if num_remove == 0:
        keys = list(lines_dict.keys())
        counts_arr = np.array([lines_dict[k] for k in keys], dtype=np.int64)
        score = _top_k_sum_from_array(counts_arr)
        best_mask = np.ones(n, dtype=bool)
        return score, base_points[best_mask]
    
    keys = list(lines_dict.keys())
    counts_arr = np.array([lines_dict[k] for k in keys], dtype=np.int64)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    if num_remove == 1:
        for i in range(n):
            if time.time() > deadline:
                break
            temp_counts = counts_arr.copy()
            for key in point_lines[i]:
                ki = key_to_idx.get(key)
                if ki is not None:
                    temp_counts[ki] -= 1
            score = _top_k_sum_from_array(temp_counts)
            if score > best_score:
                best_score = score
                best_mask = np.ones(n, dtype=bool)
                best_mask[i] = False
    elif num_remove == 2:
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                temp_counts = counts_arr.copy()
                for key in point_lines[i]:
                    ki = key_to_idx.get(key)
                    if ki is not None:
                        temp_counts[ki] -= 1
                for key in point_lines[j]:
                    ki = key_to_idx.get(key)
                    if ki is not None:
                        temp_counts[ki] -= 1
                score = _top_k_sum_from_array(temp_counts)
                if score > best_score:
                    best_score = score
                    best_mask = np.ones(n, dtype=bool)
                    best_mask[i] = False
                    best_mask[j] = False
    elif num_remove <= 5:
        # Greedy removal for larger counts
        mask = np.ones(n, dtype=bool)
        for _ in range(num_remove):
            if time.time() > deadline:
                break
            current_pts = base_points[mask]
            cur_lines, cur_point_lines = _get_all_lines_and_counts(current_pts)
            cur_keys = list(cur_lines.keys())
            cur_counts = np.array([cur_lines[k] for k in cur_keys], dtype=np.int64)
            cur_key_to_idx = {k: i for i, k in enumerate(cur_keys)}
            
            best_removal = -1
            best_new_score = -1
            for i in range(len(current_pts)):
                temp_counts = cur_counts.copy()
                for key in cur_point_lines[i]:
                    ki = cur_key_to_idx.get(key)
                    if ki is not None:
                        temp_counts[ki] -= 1
                score = _top_k_sum_from_array(temp_counts)
                if score > best_new_score:
                    best_new_score = score
                    best_removal = i
            
            if best_removal >= 0:
                mask_indices = np.where(mask)[0]
                mask[mask_indices[best_removal]] = False
            else:
                break
        
        if mask.sum() == n - num_remove:
            score, _ = _evaluate_removal(base_points[mask], 0, deadline)
            if score > best_score:
                best_score = score
                best_mask = mask
    
    if best_mask is None:
        return -1, None
    return best_score, base_points[best_mask]

def _try_grid_with_removal(a, b, num_remove, deadline):
    """Try a grid of size a*b, removing num_remove points to get 54."""
    if a * b != 54 + num_remove:
        return None, None
    pts_base = _generate_grid(a, b)
    score, pts_best = _evaluate_removal(pts_base, num_remove, deadline)
    if pts_best is not None and score > 0:
        lines_dict, _ = _get_all_lines_and_counts(pts_best)
        lines_arr = _select_top_lines(lines_dict)
        return pts_best, lines_arr
    return None, None

def _local_search_points(points, deadline):
    """
    Perform local search on point positions to improve top-27 line incidence sum.
    """
    n = points.shape[0]
    pts = points.copy()
    
    lines_dict, point_lines = _get_all_lines_and_counts(pts)
    keys_list = list(lines_dict.keys())
    counts_arr = np.array([lines_dict[k] for k in keys_list], dtype=np.int64)
    key_to_idx = {k: i for i, k in enumerate(keys_list)}
    
    current_score = _top_k_sum_from_array(counts_arr)
    improved = True
    max_iter = 0
    
    while improved and time.time() < deadline - 30 and max_iter < 50:
        improved = False
        max_iter += 1
        
        # Recompute if we made a move
        if max_iter > 1:
            lines_dict, point_lines = _get_all_lines_and_counts(pts)
            keys_list = list(lines_dict.keys())
            counts_arr = np.array([lines_dict[k] for k in keys_list], dtype=np.int64)
            key_to_idx = {k: i for i, k in enumerate(keys_list)}
        
        for idx in range(n):
            if time.time() > deadline - 30:
                break
            
            x0, y0 = int(pts[idx, 0]), int(pts[idx, 1])
            best_move = None
            best_move_score = current_score
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x0 + dx, y0 + dy
                    if new_x < 1 or new_y < 1:
                        continue
                    if np.any((pts[:, 0] == new_x) & (pts[:, 1] == new_y)):
                        continue
                    
                    temp_counts = counts_arr.copy()
                    
                    for key in point_lines[idx]:
                        ki = key_to_idx.get(key)
                        if ki is not None:
                            temp_counts[ki] -= 1
                    
                    new_key_set = set()
                    for j in range(n):
                        if j == idx:
                            continue
                        dx2 = int(pts[j, 0]) - new_x
                        dy2 = int(pts[j, 1]) - new_y
                        A = dy2
                        B = -dx2
                        C = dx2 * new_y - dy2 * new_x
                        key = _normalize_line(A, B, C)
                        if key is not None:
                            new_key_set.add(key)
                    
                    for key in new_key_set:
                        ki = key_to_idx.get(key)
                        if ki is not None:
                            temp_counts[ki] += 1
                        else:
                            temp_counts = np.append(temp_counts, 1)
                            key_to_idx[key] = len(temp_counts) - 1
                    
                    score = _top_k_sum_from_array(temp_counts)
                    if score > best_move_score:
                        best_move_score = score
                        best_move = (new_x, new_y)
            
            if best_move is not None:
                pts[idx] = best_move
                current_score = best_move_score
                improved = True
                break
    
    return pts, current_score

def _local_search_lines(points, lines, deadline):
    """
    Perform local search on line selection to maximize total incidence.
    """
    lns = lines.copy()
    current_score = _compute_incidence(points, lns)
    lines_dict, _ = _get_all_lines_and_counts(points)
    
    improved = True
    max_iter = 0
    
    while improved and time.time() < deadline - 30 and max_iter < 100:
        improved = False
        max_iter += 1
        
        current_keys = set()
        for j in range(27):
            key = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            if key:
                current_keys.add(key)
        
        sorted_candidates = sorted(lines_dict.items(), key=lambda x: -x[1])
        candidates = [(k, c) for k, c in sorted_candidates if k not in current_keys][:15]
        
        for idx in range(27):
            if time.time() > deadline - 30:
                break
            
            best_swap = None
            best_swap_score = current_score
            
            for key, count in candidates:
                new_lines = lns.copy()
                new_lines[idx] = key
                new_score = _compute_incidence(points, new_lines)
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = key
            
            if best_swap is not None:
                lns[idx] = best_swap
                current_score = best_swap_score
                improved = True
                break
    
    return lns, current_score

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, lines, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lines
    
    # Phase 1: Exact grids
    exact_grids = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    for a, b in exact_grids:
        if time.time() > deadline - 180:
            break
        pts, lines = _try_exact_grid(a, b)
        if pts is not None:
            I = _compute_incidence(pts, lines)
            update_best(pts, lines, I)
    
    # Phase 2: Near-rectangular grids with removals
    grid_configs = [
        (7, 8, 2), (8, 7, 2), (5, 11, 1), (11, 5, 1),
        (4, 14, 2), (14, 4, 2), (3, 19, 3), (19, 3, 3)
    ]
    for a, b, rem in grid_configs:
        if time.time() > deadline - 180:
            break
        pts, lines = _try_grid_with_removal(a, b, rem, deadline)
        if pts is not None:
            I = _compute_incidence(pts, lines)
            update_best(pts, lines, I)
    
    # Phase 3: Larger grids with greedy removal
    large_grids = [(6, 10, 6), (10, 6, 6), (5, 12, 6), (12, 5, 6)]
    for a, b, rem in large_grids:
        if time.time() > deadline - 150:
            break
        pts_base = _generate_grid(a, b)
        score, pts_best = _evaluate_removal(pts_base, rem, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = _compute_incidence(pts_best, lines_arr)
            update_best(pts_best, lines_arr, I)
    
    # Phase 4: Local search on points
    if best_points is not None and time.time() < deadline - 90:
        refined_pts, refined_score = _local_search_points(best_points, deadline)
        if refined_score > best_score:
            lines_dict, _ = _get_all_lines_and_counts(refined_pts)
            lines_arr = _select_top_lines(lines_dict)
            I = _compute_incidence(refined_pts, lines_arr)
            update_best(refined_pts, lines_arr, I)
    
    # Phase 5: Local search on lines
    if best_points is not None and best_lines is not None and time.time() < deadline - 60:
        refined_lines, refined_score = _local_search_lines(best_points, best_lines, deadline)
        if refined_score > best_score:
            update_best(best_points, refined_lines, refined_score)
    
    # Phase 6: More local search iterations
    if best_points is not None and time.time() < deadline - 60:
        for _ in range(2):
            if time.time() > deadline - 30:
                break
            refined_pts, refined_score = _local_search_points(best_points, deadline)
            if refined_score > best_score:
                lines_dict, _ = _get_all_lines_and_counts(refined_pts)
                lines_arr = _select_top_lines(lines_dict)
                I = _compute_incidence(refined_pts, lines_arr)
                update_best(refined_pts, lines_arr, I)
            else:
                break
    
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = []
        for c in range(1, 7):
            lines_list.append((1, 0, -c))
        for c in range(1, 10):
            lines_list.append((0, 1, -c))
        for c in [0, 1, 2, 3]:
            lines_list.append((1, -1, c))
        for c in [7, 8, 9, 10]:
            lines_list.append((1, 1, -c))
        for c in [4, -1]:
            lines_list.append((1, -1, c))
        for c in [6, 11]:
            lines_list.append((1, 1, -c))
        best_lines = np.array(lines_list, dtype=np.int64)
    else:
        best_lines = _select_top_lines(_get_all_lines_and_counts(best_points)[0])
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
