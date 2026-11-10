# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _gcd3(a, b, c):
    a, b, c = abs(a), abs(b), abs(c)
    if a == 0 and b == 0 and c == 0:
        return 1
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.count_nonzero(vals == 0)
    return int(total)

def _validate(points, lines):
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(np.isfinite(lines)):
        return False
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != 54:
        return False
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
    norm_lines = set()
    for A, B, C in lines:
        nl = _normalize_line(int(A), int(B), int(C))
        if nl in norm_lines:
            return False
        norm_lines.add(nl)
    return True

def _generate_grid_points(rows, cols):
    xs = np.arange(1, rows + 1)
    ys = np.arange(1, cols + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _get_candidate_lines(points):
    """
    Generates a list of candidate lines with their incidence counts.
    Focuses on vertical, horizontal, and slope +/- 1 diagonals.
    """
    n = len(points)
    line_to_count = {}
    
    # Verticals
    unique_xs = np.unique(points[:, 0])
    for x in unique_xs:
        A, B, C = 1, 0, -x
        nl = _normalize_line(A, B, C)
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = int(np.count_nonzero(vals == 0))
        if nl not in line_to_count:
            line_to_count[nl] = count
            
    # Horizontals
    unique_ys = np.unique(points[:, 1])
    for y in unique_ys:
        A, B, C = 0, 1, -y
        nl = _normalize_line(A, B, C)
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = int(np.count_nonzero(vals == 0))
        if nl not in line_to_count:
            line_to_count[nl] = count
            
    # Diagonals (Slope 1 and -1)
    # Slope 1: y - x = c => x - y + c = 0? No, x - y = -c => (1, -1, c)
    # Slope -1: y = -x + c => x + y - c = 0 => (1, 1, -c)
    
    # Generate all pairs to find diagonals efficiently?
    # For small grids, iterating pairs is fine.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Slope 1
            if x2 - x1 == y2 - y1:
                # y - x = y1 - x1 => x - y + (y1 - x1) = 0
                c = y1 - x1
                A, B, C = 1, -1, c
                nl = _normalize_line(A, B, C)
                if nl not in line_to_count:
                    vals = A * points[:, 0] + B * points[:, 1] + C
                    count = int(np.count_nonzero(vals == 0))
                    line_to_count[nl] = count
                    
            # Slope -1
            elif x2 - x1 == -(y2 - y1):
                # x + y = x1 + y1 => x + y - (x1 + y1) = 0
                c = x1 + y1
                A, B, C = 1, 1, -c
                nl = _normalize_line(A, B, C)
                if nl not in line_to_count:
                    vals = A * points[:, 0] + B * points[:, 1] + C
                    count = int(np.count_nonzero(vals == 0))
                    line_to_count[nl] = count
                    
    return sorted(line_to_count.items(), key=lambda x: -x[1])

def _optimize_lines(points, candidate_lines, num_lines=27, time_limit=10.0):
    """
    Selects the best num_lines from candidate_lines using local search.
    """
    if len(candidate_lines) < num_lines:
        return [l for l, c in candidate_lines], _count_incidents(points, [l for l, c in candidate_lines])
        
    # Initialize with top num_lines
    initial_indices = list(range(num_lines))
    current_set = set(initial_indices)
    current_inc = sum(candidate_lines[i][1] for i in current_set)
    # Note: This sum is an upper bound if lines are disjoint, but we need exact count.
    current_lines = [candidate_lines[i][0] for i in current_set]
    current_inc = _count_incidents(points, current_lines)
    
    best_set = set(current_set)
    best_inc = current_inc
    
    line_arrays = np.array([candidate_lines[i][0] for i in range(len(candidate_lines))], dtype=np.int64)
    inc_matrix = (line_arrays[:, 0:1] * points[:, 0:1].T + 
                  line_arrays[:, 1:2] * points[:, 1:2].T + 
                  line_arrays[:, 2:3]) == 0
    line_sums = np.array([candidate_lines[i][1] for i in range(len(candidate_lines))], dtype=np.int64)
    
    in_current = np.zeros(len(candidate_lines), dtype=bool)
    for idx in current_set:
        in_current[idx] = True
        
    start = time.time()
    T = 10.0
    cooling = 0.999
    
    while T > 0.01 and time.time() - start < time_limit:
        i = list(current_set)[np.random.randint(len(current_set))]
        j = np.random.randint(len(candidate_lines))
        if in_current[j]:
            continue
            
        # Calculate delta
        # Remove i, add j
        # Approximate delta: line_sums[j] - line_sums[i]
        # But interactions matter.
        
        new_set = set(current_set)
        new_set.discard(i)
        new_set.add(j)
        
        # Quick check: if estimated improvement is negative, maybe skip?
        # But we need exact count for accuracy.
        
        new_lines = [candidate_lines[k][0] for k in new_set]
        new_inc = _count_incidents(points, new_lines)
        
        delta = new_inc - current_inc
        
        if delta > 0 or (np.random.random() < math.exp(delta / T)):
            current_set = new_set
            current_inc = new_inc
            in_current[i] = False
            in_current[j] = True
            if current_inc > best_inc:
                best_set = set(current_set)
                best_inc = current_inc
                
        T *= cooling
        
    best_lines = [candidate_lines[i][0] for i in best_set]
    return best_lines, best_inc

def _remove_points_to_fit(points, lines, target_count=54):
    """
    Removes points to reduce count to target_count, minimizing loss of incidence.
    """
    current_points = points
    while len(current_points) > target_count:
        # Calculate incidence count per point
        inc_per_point = np.zeros(len(current_points), dtype=np.int64)
        for A, B, C in lines:
            vals = A * current_points[:, 0] + B * current_points[:, 1] + C
            inc_per_point += (vals == 0).astype(np.int64)
            
        # Find point with minimum incidence
        min_idx = np.argmin(inc_per_point)
        current_points = np.delete(current_points, min_idx, axis=0)
        
    return current_points

def _try_geometry(rows, cols, max_time=50.0):
    points = _generate_grid_points(rows, cols)
    n_pts = len(points)
    
    if n_pts < 54:
        return None
        
    candidate_lines = _get_candidate_lines(points)
    
    if len(candidate_lines) < 27:
        return None
        
    # If we have more than 54 points, we will remove some later.
    # First, optimize lines on the FULL set to find good lines.
    lines, inc_full = _optimize_lines(points, candidate_lines, 27, time_limit=max_time/2)
    
    # If we need to remove points, do so.
    if n_pts > 54:
        # We want to keep points that are on many of the selected lines.
        # So we remove points with fewest incidences on the selected lines.
        final_points = _remove_points_to_fit(points, lines, 54)
        final_inc = _count_incidents(final_points, lines)
    else:
        final_points = points
        final_inc = inc_full
        
    return final_points, lines, final_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    max_time = 590.0
    
    best_result = None
    best_score = -1
    
    geometries = [
        (6, 9),
        (9, 6),
        (7, 8), # 56 points, remove 2
        (8, 7), # 56 points, remove 2
        (5, 11), # 55 points, remove 1
        (11, 5), # 55 points, remove 1
        (3, 18), # 54 points
        (18, 3), # 54 points
    ]
    
    for rows, cols in geometries:
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:
            break
        time_budget = max_time - 10 - elapsed
        if time_budget < 20:
            break
            
        try:
            result = _try_geometry(rows, cols, max_time=time_budget)
            if result is not None:
                points, lines, score = result
                if score > best_score:
                    best_score = score
                    best_result = (points, lines)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to 6x9
        points, lines, _ = _try_geometry(6, 9, max_time=10)
        best_result = (points, lines)
        
    points, lines = best_result
    
    if not _validate(points, lines):
        # Fallback to simple grid
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines.append((1, 1, -c))
        lines = np.array(lines, dtype=np.int64)
        
    return points, lines

# EVOLVE-BLOCK-END
