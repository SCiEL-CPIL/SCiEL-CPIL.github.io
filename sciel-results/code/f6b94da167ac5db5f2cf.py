import numpy as np
from math import gcd
import time
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count total incidences between points and lines."""
    total = 0
    for A, B, C in lines:
        for x, y in points:
            if A * x + B * y + C == 0:
                total += 1
    return total

def _find_best_lines_for_points(points, num_lines=27, time_limit=None, start_time=None):
    """Find the best num_lines lines for a given set of points."""
    if time_limit is None:
        time_limit = 580
    if start_time is None:
        start_time = time.time()
    
    n = len(points)
    lines_set = set()
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            lines_set.add(_normalize_line(A, B, C))
    
    if time.time() - start_time > time_limit:
        return None
    
    line_list = list(lines_set)
    incidences = []
    for A, B, C in line_list:
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        incidences.append(count)
    
    line_indices = sorted(range(len(line_list)), key=lambda k: incidences[k], reverse=True)
    selected = [line_list[idx] for idx in line_indices[:num_lines]]
    
    return selected

def _make_grid(m, n, offset=(0, 0)):
    """Create an m x n grid of points."""
    xs = np.arange(m) + offset[0]
    ys = np.arange(n) + offset[1]
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _try_configuration(points, num_lines=27, time_limit=580, start_time=None):
    """Try a point configuration and return (score, points, lines) or None."""
    if start_time is None:
        start_time = time.time()
    if time.time() - start_time > time_limit:
        return None
    
    lines = _find_best_lines_for_points(points, num_lines, time_limit, start_time)
    if lines is None:
        return None
    
    score = _count_incidences(points, lines)
    return (score, points, np.array(lines, dtype=np.int64))

def _local_search(points, lines, num_lines=27, max_iter=50, time_limit=580, start_time=None):
    """Local search: try moving individual points to improve incidence count."""
    if start_time is None:
        start_time = time.time()
    if time.time() - start_time > time_limit:
        return points, lines, _count_incidences(points, lines)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidences(best_points, best_lines)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(len(best_points)):
            if time.time() - start_time > time_limit:
                break
            
            orig_x, orig_y = best_points[i]
            # Try moving this point to nearby positions
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x, new_y = orig_x + dx, orig_y + dy
                    trial_points = best_points.copy()
                    trial_points[i] = [new_x, new_y]
                    
                    # Check for duplicate points
                    if len(set(map(tuple, trial_points.tolist()))) != len(trial_points):
                        continue
                    
                    trial_lines = _find_best_lines_for_points(
                        trial_points, num_lines, time_limit, start_time)
                    if trial_lines is None:
                        continue
                    
                    trial_score = _count_incidences(trial_points, trial_lines)
                    if trial_score > best_score:
                        best_points = trial_points
                        best_lines = np.array(trial_lines, dtype=np.int64)
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    time_limit = 580
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Try various grid configurations
    # Format: (m, n, points_to_remove, description)
    configs = [
        # Exact 54-point grids
        (6, 9, [], "6x9 grid"),
        (9, 6, [], "9x6 grid"),
        
        # 56-point grids with 2 removed
        (7, 8, [0, 55], "7x8 remove corners"),
        (7, 8, [0, 54], "7x8 remove (1,1) and (7,7)"),
        (7, 8, [0, 53], "7x8 remove (1,1) and (6,8)"),
        (7, 8, [5, 50], "7x8 remove (1,6) and (6,7)"),
        (7, 8, [10, 45], "7x8 remove (2,6) and (6,6)"),
        (7, 8, [2, 53], "7x8 remove (1,3) and (6,8)"),
        (8, 7, [0, 55], "8x7 remove corners"),
        (8, 7, [0, 54], "8x7 remove (1,1) and (7,7)"),
        (8, 7, [5, 50], "8x7 remove (1,6) and (6,7)"),
        
        # 55-point grids with 1 removed
        (5, 11, [0], "5x11 remove (1,1)"),
        (5, 11, [54], "5x11 remove (5,11)"),
        (5, 11, [10], "5x11 remove (2,1)"),
        (11, 5, [0], "11x5 remove (1,1)"),
        (11, 5, [54], "11x5 remove (11,5)"),
        
        # 57-point grids with 3 removed
        (9, 7, [0, 55, 56], "9x7 remove 3 corners"),
        (9, 7, [0, 1, 56], "9x7 remove top row ends and bottom right"),
    ]
    
    for m, n, remove_idx, desc in configs:
        if time.time() - start_time > time_limit:
            break
        
        grid = _make_grid(m, n)
        
        if remove_idx:
            points = np.delete(grid, remove_idx, axis=0)
        else:
            points = grid
        
        if len(points) != 54:
            continue
        
        # Check for duplicates
        if len(set(map(tuple, points.tolist()))) != 54:
            continue
        
        result = _try_configuration(points, 27, time_limit, start_time)
        if result is not None:
            score, pts, lns = result
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_lines = lns.copy()
    
    # Local search from best configuration
    if best_points is not None and time.time() - start_time < time_limit - 30:
        best_points, best_lines, best_score = _local_search(
            best_points, best_lines, 27, max_iter=30, 
            time_limit=time_limit, start_time=start_time)
    
    # Final safety check
    if time.time() - start_time > 590:
        # Return whatever we have
        pass
    
    if best_points is None:
        # Fallback: simple 6x9 grid
        best_points = _make_grid(6, 9)
        best_lines = _find_best_lines_for_points(best_points, 27, 5, start_time)
        if best_lines is None:
            # Emergency fallback
            best_lines = np.zeros((27, 3), dtype=np.int64)
            for i in range(27):
                best_lines[i] = [0, 1, -i]  # horizontal lines
    
    return best_points, best_lines
