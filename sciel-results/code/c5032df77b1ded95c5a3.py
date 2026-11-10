import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_grid(cols, rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    line_set = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            line_set.add(norm)
    lines_list = list(line_set)
    # Vectorized count for all lines
    if not lines_list:
         return points, np.zeros((27, 3), dtype=np.int64)
    lines_arr = np.array(lines_list, dtype=np.int64)
    A = lines_arr[:, 0:1]
    B = lines_arr[:, 1:2]
    C = lines_arr[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    counts = np.count_nonzero(val == 0, axis=1)
    top = np.argsort(-counts)[:27]
    lines = lines_arr[top]
    return points, lines

def _anneal(points, lines, deadline, seed=0):
    points = points.copy()
    lines = lines.copy()
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    M = (A * x + B * y + C == 0)
    score = int(np.sum(M))
    best_score = score
    best_points = points.copy()
    best_lines = lines.copy()
    
    line_norms = set()
    for i in range(27):
        line_norms.add(_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])))
        
    rng = np.random.default_rng(seed)
    T = 5.0  # Higher initial temperature
    cool = 0.99995  # Slightly faster cooling
    point_set = set(map(tuple, points.tolist()))
    
    # Pre-allocate some arrays for speed
    new_col_arr = np.zeros(27, dtype=np.int64)
    new_row_arr = np.zeros(54, dtype=np.int64)
    
    stagnation_counter = 0
    max_stagnation = 1000
    
    while time.time() < deadline:
        if rng.random() < 0.7:
            # Point move
            pi = int(rng.integers(0, 54))
            ox, oy = points[pi]
            old_col = int(np.count_nonzero(M[:, pi]))
            
            # Try a few random moves
            moved = False
            for _ in range(15):
                dx = int(rng.integers(-6, 7))
                dy = int(rng.integers(-6, 7))
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if (nx, ny) in point_set:
                    continue
                
                # Calculate new incidences for this point against all lines
                new_col_arr = A.ravel() * nx + B.ravel() * ny + C.ravel()
                new_col = int(np.count_nonzero(new_col_arr == 0))
                delta = new_col - old_col
                
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    points[pi] = [nx, ny]
                    M[:, pi] = (new_col_arr == 0).astype(bool)
                    score += delta
                    
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    moved = True
                    break
            
            if not moved:
                 stagnation_counter += 1
                 
        else:
            # Line move
            li = int(rng.integers(0, 27))
            p1 = int(rng.integers(0, 54))
            p2 = int(rng.integers(0, 54))
            if p1 == p2:
                continue
            x1, y1 = points[p1]
            x2, y2 = points[p2]
            A2 = y2 - y1
            B2 = x1 - x2
            C2 = x2 * y1 - x1 * y2
            if A2 == 0 and B2 == 0:
                continue
            
            norm = _normalize_line(A2, B2, C2)
            
            # Check for duplicates
            # We need to check if this normalized line already exists in the current set (excluding li)
            # Since we are replacing li, we check if norm is in line_norms.
            # If it is, it's a duplicate. But if it's the same as the current line li, it's not a "new" duplicate in the sense of breaking uniqueness, 
            # but it's also not an improvement. 
            # Actually, if we pick the same line, delta is 0. 
            # If we pick a line that exists at another index, it's invalid.
            
            # Efficient check: 
            # If norm == current norm of li, delta is 0 (no change).
            # If norm != current norm of li, we must ensure norm is not in the set of other lines.
            
            current_norm = _normalize_line(int(lines[li][0]), int(lines[li][1]), int(lines[li][2]))
            
            if norm == current_norm:
                continue # No change
            
            # Check if norm is in the set of other lines
            # We can't just check `norm in line_norms` because line_norms contains current_norm.
            # If norm is in line_norms, it must be one of the others (since norm != current_norm).
            if norm in line_norms:
                continue
            
            new_row_arr = A2 * points[:, 0] + B2 * points[:, 1] + C2
            new_row = int(np.count_nonzero(new_row_arr == 0))
            old_row = int(np.count_nonzero(M[li, :]))
            delta = new_row - old_row
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                # Update line
                lines[li] = norm
                line_norms.discard(current_norm)
                line_norms.add(norm)
                M[li, :] = (new_row_arr == 0).astype(bool)
                score += delta
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
        T *= cool
        
        # Restart mechanism if stuck
        if stagnation_counter > max_stagnation:
            T = max(T, 2.0) # Increase temperature
            stagnation_counter = 0
            # Optionally, revert to best? No, that's too aggressive. Just keep exploring.
            # Or, restart from best?
            # Let's try a soft restart: reset T
            T = 3.0

    return best_points, best_lines, best_score

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    ls = set()
    for i in range(27):
        A, B, C = int(lines[i][0]), int(lines[i][1]), int(lines[i][2])
        if A == 0 and B == 0:
            return False
        n = _normalize_line(A, B, C)
        if n in ls:
            return False
        ls.add(n)
    return True

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 590
    
    # Try different grid configurations
    # 9x6 is usually very strong.
    # We will run annealing on the best grid.
    
    configs = [(9, 6), (6, 9), (18, 3), (3, 18), (27, 2)]
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # First, find the best initial grid score
    initial_candidates = []
    for c, r in configs:
        pts, lns = _build_grid(c, r)
        s = int(np.sum((lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]) == 0))
        initial_candidates.append((s, c, r, pts, lns))
        
    initial_candidates.sort(key=lambda x: -x[0])
    
    # Allocate time to the top 2-3 configurations
    # Give the best one most time.
    time_per_config = 150
    if len(initial_candidates) > 1:
        time_per_config = 100
    if len(initial_candidates) > 2:
        time_per_config = 80
        
    for idx, (s, c, r, pts, lns) in enumerate(initial_candidates[:3]):
        if time.time() > deadline - 20:
            break
            
        # If the initial score is already very high, we might not need to anneal as long, 
        # but annealing usually improves it significantly.
        
        # Calculate deadline for this anneal
        # Reserve 10s for next config if any
        next_configs = len(initial_candidates) - 1 - idx
        time_left_for_this = min(time_per_config, deadline - time.time() - (next_configs * 10))
        
        if time_left_for_this < 5:
            continue
            
        anneal_deadline = time.time() + time_left_for_this
        
        bp, bl, bs = _anneal(pts, lns, anneal_deadline, seed=42 + idx * 1000)
        
        if bs > best_score:
            best_score = bs
            best_points = bp.copy()
            best_lines = bl.copy()
            
    if best_points is None:
        # Fallback to the best initial
        s, c, r, pts, lns = initial_candidates[0]
        best_points = pts
        best_lines = lns
        best_score = s

    if not _is_valid(best_points, best_lines):
        # Fallback to a simple valid grid
        pts, lns = _build_grid(9, 6)
        best_points = pts
        best_lines = lns

    return best_points, best_lines
