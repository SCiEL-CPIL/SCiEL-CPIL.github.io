import numpy as np
import time
from math import gcd
from itertools import combinations

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

def _get_all_lines_and_counts(points):
    """
    Returns a dictionary: line_key -> incidence_count
    and a list: point_lines[i] = set of line_keys passing through point i
    """
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
    
    # Count incidences for each line
    # Optimization: We can just count how many times each key appears in the pair loop?
    # No, a line with k points has C(k,2) pairs.
    # It's faster to compute the line equation for all pairs, normalize, and then
    # for each unique line, count points.
    # But we already have point_lines.
    # We can count incidences by iterating points and checking lines?
    # No, we have the lines.
    # For each line key, we need to count how many points satisfy A*x+B*y+C=0.
    # This is O(N * L). L is number of unique lines. L ~ N^2/2.
    # N=54, L~1500. 54*1500 = 81,000 ops. Very fast.
    
    for key in lines_dict:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
        
    return lines_dict, point_lines

def _evaluate_configuration(points):
    """
    Returns the sum of incidences of the top 27 lines.
    """
    lines_dict, _ = _get_all_lines_and_counts(points)
    if not lines_dict:
        return 0
    counts = list(lines_dict.values())
    counts.sort(reverse=True)
    k = min(27, len(counts))
    return sum(counts[:k])

def _get_top_lines(points, k=27):
    lines_dict, _ = _get_all_lines_and_counts(points)
    if not lines_dict:
        # Fallback
        return np.zeros((k, 3), dtype=np.int64)
    
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    
    if len(selected) < k:
        # Pad with arbitrary lines if not enough
        # This shouldn't happen with 54 points
        all_keys = list(lines_dict.keys())
        sel_keys = set(selected)
        pad_keys = [key for key in all_keys if key not in sel_keys]
        while len(pad_keys) < k - len(selected):
            pad_keys.append((1, 0, 1000000 + len(pad_keys)))
        selected.extend(pad_keys[:k - len(selected)])
        
    return np.array(selected, dtype=np.int64)

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _local_search(points, max_rounds=10, time_budget=300):
    """
    Local search by moving points to nearby empty integer coordinates.
    """
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    start_time = time.time()
    
    for r in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        n = len(current_points)
        occupied = set(map(tuple, current_points))
        
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            
            orig_pt = current_points[i]
            # Search radius 3
            best_move = None
            best_score = current_score
            
            # Generate candidates in a spiral or box
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (int(orig_pt[0] + dx), int(orig_pt[1] + dy))
                    if cand in occupied:
                        continue
                    
                    # Try moving point i to cand
                    trial_points = current_points.copy()
                    trial_points[i] = cand
                    
                    # Quick check: does this create duplicates? (Handled by occupied set)
                    # Evaluate
                    trial_score = _evaluate_configuration(trial_points)
                    
                    if trial_score > best_score:
                        best_score = trial_score
                        best_move = cand
            
            if best_move is not None:
                # Update state
                old_pt = tuple(current_points[i])
                occupied.remove(old_pt)
                current_points[i] = best_move
                occupied.add(best_move)
                current_score = best_score
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
    
    # Grid configurations (a, b) such that a*b >= 54
    # We will try to remove points if a*b > 54
    grid_configs = [
        (6, 9), (9, 6),
        (7, 8), (8, 7),
        (5, 11), (11, 5),
        (4, 14), (14, 4),
        (3, 18), (18, 3),
        (2, 27), (27, 2),
        (1, 54), (54, 1),
        (7, 7), # 49, too small? No, we need 54. So 7x8 is better.
        (8, 8), # 64, remove 10
        (9, 9), # 81, remove 27
    ]
    
    # Filter valid grids (size >= 54)
    valid_grids = []
    for a, b in grid_configs:
        if a * b >= 54:
            valid_grids.append((a, b))
    
    # Sort by "compactness" or just iterate
    for a, b in valid_grids:
        if time.time() > deadline - 60:
            break
            
        pts = _generate_grid(a, b)
        
        if len(pts) > 54:
            # Remove points
            # Strategy: Remove points that are "least central" or on the boundary
            # For now, just remove the last points (corners)
            # A better way: Randomly remove or use a heuristic.
            # Let's try removing corners first.
            # In a grid generated by meshgrid, indices are row-major.
            # Corners are (0,0), (0,-1), (-1,0), (-1,-1) in 0-based index of the grid.
            # But we need to be careful which points to remove.
            
            # Heuristic: Points with fewer neighbors in the grid are less valuable?
            # Or just remove points from the "periphery".
            
            # Simple approach: Try removing each point one by one and pick the best sequence?
            # Too slow.
            
            # Let's just take the first 54 points? No, that's biased.
            # Let's take points in a spiral from the center?
            # Or just use the local search to fix it.
            # Initialize with the full grid, then local search will move points?
            # No, local search keeps 54 points.
            
            # We need to reduce to 54 points.
            # Let's try removing the 4 corners of the grid.
            # For a x b grid:
            # Indices: 0, b-1, a*b-b, a*b-1 are corners.
            # If we need to remove R points:
            R = len(pts) - 54
            remove_indices = []
            
            # Prioritize corners, then edges
            # This is complex. Let's just remove the last R points from the array?
            # The array is filled row by row. So last points are bottom-right.
            # This creates a jagged edge.
            
            # Better: Remove points that are far from the center of mass?
            center_x = (a + 1) / 2.0
            center_y = (b + 1) / 2.0
            distances = np.sqrt((pts[:, 0] - center_x)**2 + (pts[:, 1] - center_y)**2)
            # Sort by distance descending
            order = np.argsort(distances)[::-1]
            remove_indices = order[:R]
            
            mask = np.ones(len(pts), dtype=bool)
            mask[remove_indices] = False
            pts = pts[mask]
        
        if len(pts) != 54:
            continue
            
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
        # Local search on this configuration
        if time.time() < deadline - 30:
            new_score, new_pts = _local_search(pts, max_rounds=5, time_budget=min(60, deadline - time.time() - 10))
            if new_score > best_score:
                update_best(new_pts, new_score)
    
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    return best_points, best_lines
