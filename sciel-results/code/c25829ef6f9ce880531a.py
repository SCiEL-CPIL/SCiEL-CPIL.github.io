# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    # Compute gcd
    g = abs(A)
    for v in [abs(B), abs(C)]:
        while v:
            A, v = v, A % v
        g = A
    if g == 0:
        return None
    return (A // g, B // g, C // g)

def _line_from_points(p1, p2):
    """Get normalized line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return None
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    count = 0
    for p in points:
        x, y = p
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count

def _validate(points, lines):
    """Check validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check for NaN/inf
    if not np.all(np.isfinite(points.astype(float))):
        return False
    if not np.all(np.isfinite(lines.astype(float))):
        return False
    # Check integer-ness
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Check repeated points
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key in seen:
            return False
        seen.add(key)
    # Check lines
    seen_lines = set()
    for A, B, C in lines:
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _initial_star_config():
    """Create a star configuration: 27 lines through origin, points on lines."""
    points = [(0, 0)]  # Center
    lines = []
    # Use 27 different slopes through origin
    # Line: y = m*x -> m*x - y = 0 -> (m, -1, 0)
    # Also vertical: x = c -> (1, 0, -c)
    slopes = list(range(1, 27))  # 26 slopes
    # Add vertical line x=0: (1, 0, 0)
    for m in slopes:
        lines.append((m, -1, 0))
    lines.append((1, 0, 0))
    
    # Place 53 more points on these lines
    # For each line, place points at various distances
    # Line (m, -1, 0): points (x, m*x) for x = 1, 2, ...
    # Line (1, 0, 0): points (0, y) for y = 1, 2, ...
    
    used = set([(0, 0)])
    for i, (A, B, C) in enumerate(lines):
        # Place 2 points on each line (53 = 27*2 - 1, one line gets 1)
        n_pts = 2 if i < 26 else 1
        if A != 0 and B != 0:
            # Line: A*x + B*y + C = 0, C=0, so y = -(A/B)*x
            # For (m, -1, 0): y = m*x
            m_val = A // B if B != 0 else 0
            # Actually let's just parametrize
            # A*x + B*y = 0 -> y = -A*x/B
            # For (m, -1, 0): m*x - y = 0 -> y = m*x
            for t in range(1, n_pts + 1):
                x = t
                y = -A * x // B if B != 0 else 0
                pt = (x, y)
                if pt not in used:
                    points.append(pt)
                    used.add(pt)
        elif A != 0:
            # Vertical line: A*x + C = 0 -> x = -C/A
            x = -C // A if A != 0 else 0
            for t in range(1, n_pts + 1):
                y = t
                pt = (x, y)
                if pt not in used:
                    points.append(pt)
                    used.add(pt)
    
    # Pad if needed
    counter = 1
    while len(points) < 54:
        pt = (counter, counter + 100)
        if pt not in used:
            points.append(pt)
            used.add(pt)
        counter += 1
    
    points = np.array(points[:54], dtype=np.int64)
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def _initial_grid_config():
    """Classic Szemeredi-Trotter grid."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines

def _initial_pencil_grid():
    """Hybrid: some points in a grid, lines through pairs."""
    # Start with a 9x6 grid = 54 points
    points = []
    for x in range(1, 10):
        for y in range(1, 7):
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    # Find lines with many points
    # For a 9x6 grid, horizontal lines have 9 points, vertical have 6
    # Diagonals have fewer
    # Let's pick 27 lines that maximize incidences
    # Horizontal: 6 lines, each with 9 points = 54
    # Vertical: 9 lines, each with 6 points = 54
    # Total 15 lines, 108 incidences
    # Need 12 more lines
    # Diagonals with slope 1: at most min(9,6)=6 points
    # Let's use 6 diagonal lines with slope 1 and 6 with slope -1
    
    lines = []
    # 6 horizontal lines: y = c -> (0, 1, -c)
    for c in range(1, 7):
        lines.append((0, 1, -c))
    # 9 vertical lines: x = c -> (1, 0, -c)
    for c in range(1, 10):
        lines.append((1, 0, -c))
    # 6 diagonal lines with slope 1: y - x = c -> (-1, 1, -c)
    # For 9x6 grid, c ranges from 1-9=-8 to 6-1=5, but we want lines with >= 2 points
    # y = x + c, points where 1<=x<=9, 1<=y<=6
    # For c=-8: x=9,y=1 (1 pt)
    # For c=-7: x=8,9 y=1,2 (2 pts)
    # ...
    # Let's just pick 6 that have many points
    for c in range(-3, 3):  # c = -3,-2,-1,0,1,2
        lines.append((-1, 1, -c))
    
    lines = np.array(lines[:27], dtype=np.int64)
    return points, lines

def _local_search(points, lines, time_budget):
    """Local search to improve incidences."""
    start_time = time.time()
    current_inc = _count_incidences(points, lines)
    best_inc = current_inc
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Precompute incidence matrix
    def compute_incidence_matrix(pts, lns):
        mat = np.zeros((len(pts), len(lns)), dtype=np.int8)
        for i, p in enumerate(pts):
            x, y = p
            for j, (A, B, C) in enumerate(lns):
                if A * x + B * y + C == 0:
                    mat[i, j] = 1
        return mat
    
    inc_mat = compute_incidence_matrix(points, lines)
    current_inc = int(inc_mat.sum())
    best_inc = current_inc
    
    # Try moving individual points
    for iteration in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        # Try moving each point to nearby positions
        for i in range(54):
            if time.time() - start_time > time_budget:
                break
            
            old_x, old_y = points[i]
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                        continue
                    
                    # Check if this position is already used
                    is_dup = False
                    for j in range(54):
                        if j == i:
                            continue
                        if points[j][0] == new_x and points[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Compute new incidences for this point
                    new_inc = 0
                    for j, (A, B, C) in enumerate(lines):
                        if A * new_x + B * new_y + C == 0:
                            new_inc += 1
                    
                    old_point_inc = int(inc_mat[i].sum())
                    delta = new_inc - old_point_inc
                    
                    if delta > 0:
                        # Accept the move
                        points[i] = [new_x, new_y]
                        inc_mat[i] = 0
                        for j, (A, B, C) in enumerate(lines):
                            if A * new_x + B * new_y + C == 0:
                                inc_mat[i, j] = 1
                        current_inc += delta
                        improved = True
                        
                        if current_inc > best_inc:
                            best_inc = current_inc
                            best_points = points.copy()
                            best_lines = lines.copy()
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try swapping two points
            found_swap = False
            for i in range(54):
                if time.time() - start_time > time_budget:
                    break
                for j in range(i + 1, 54):
                    if time.time() - start_time > time_budget:
                        break
                    # Swap points i and j
                    pi, pj = points[i].copy(), points[j].copy()
                    points[i], points[j] = pj, pi
                    
                    # Recompute incidences for these two points
                    new_i_inc = 0
                    new_j_inc = 0
                    for k, (A, B, C) in enumerate(lines):
                        if A * points[i][0] + B * points[i][1] + C == 0:
                            new_i_inc += 1
                        if A * points[j][0] + B * points[j][1] + C == 0:
                            new_j_inc += 1
                    
                    old_i_inc = int(inc_mat[i].sum())
                    old_j_inc = int(inc_mat[j].sum())
                    delta = (new_i_inc + new_j_inc) - (old_i_inc + old_j_inc)
                    
                    if delta > 0:
                        # Accept swap
                        inc_mat[i] = 0
                        inc_mat[j] = 0
                        for k, (A, B, C) in enumerate(lines):
                            if A * points[i][0] + B * points[i][1] + C == 0:
                                inc_mat[i, k] = 1
                            if A * points[j][0] + B * points[j][1] + C == 0:
                                inc_mat[j, k] = 1
                        current_inc += delta
                        found_swap = True
                        
                        if current_inc > best_inc:
                            best_inc = current_inc
                            best_points = points.copy()
                            best_lines = lines.copy()
                        break
                    else:
                        # Revert
                        points[i], points[j] = pi, pj
                if found_swap:
                    break
            
            if not found_swap:
                # Try replacing a line
                found_line = False
                for j in range(27):
                    if time.time() - start_time > time_budget:
                        break
                    
                    # Try new lines through pairs of points
                    best_new_line = None
                    best_delta = 0
                    for i1 in range(54):
                        for i2 in range(i1 + 1, 54):
                            p1 = tuple(points[i1])
                            p2 = tuple(points[i2])
                            new_line = _line_from_points(p1, p2)
                            if new_line is None:
                                continue
                            
                            # Check if this line already exists
                            is_dup = False
                            for k in range(27):
                                if k == j:
                                    continue
                                if _normalize_line(int(lines[k][0]), int(lines[k][1]), int(lines[k][2])) == new_line:
                                    is_dup = True
                                    break
                            if is_dup:
                                continue
                            
                            # Compute incidences of this new line
                            new_line_inc = 0
                            for i in range(54):
                                x, y = points[i]
                                A, B, C = new_line
                                if A * x + B * y + C == 0:
                                    new_line_inc += 1
                            
                            old_line_inc = int(inc_mat[:, j].sum())
                            delta = new_line_inc - old_line_inc
                            
                            if delta > best_delta:
                                best_delta = delta
                                best_new_line = new_line
                            if time.time() - start_time > time_budget:
                                break
                        if time.time() - start_time > time_budget:
                            break
                    
                    if best_new_line is not None and best_delta > 0:
                        # Accept the line replacement
                        lines[j] = best_new_line
                        inc_mat[:, j] = 0
                        for i in range(54):
                            x, y = points[i]
                            A, B, C = best_new_line
                            if A * x + B * y + C == 0:
                                inc_mat[i, j] = 1
                        current_inc += best_delta
                        found_line = True
                        
                        if current_inc > best_inc:
                            best_inc = current_inc
                            best_points = points.copy()
                            best_lines = lines.copy()
                        break
                
                if not found_line:
                    break
    
    return best_points, best_lines, best_inc

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization with multiple initial configurations
    and local search.
    """
    start_time = time.time()
    deadline = 620  # Leave 10s margin before 630s kill
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid_config,
        _initial_pencil_grid,
        _initial_star_config,
    ]
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    for config_func in initial_configs:
        if time.time() - start_time > deadline:
            break
        
        try:
            points, lines = config_func()
        except Exception:
            continue
        
        if not _validate(points, lines):
            continue
        
        inc = _count_incidences(points, lines)
        
        # Local search with time budget
        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            if inc > best_inc:
                best_inc = inc
                best_points = points
                best_lines = lines
            continue
        
        time_budget = min(remaining - 2, 180)  # Max 3 min per config
        best_p, best_l, best_i = _local_search(points, lines, time_budget)
        
        if best_i > best_inc:
            best_inc = best_i
            best_points = best_p
            best_lines = best_l
    
    if best_points is None:
        # Fallback to grid
        best_points, best_lines = _initial_grid_config()
    
    # Final validation
    if not _validate(best_points, best_lines):
        best_points, best_lines = _initial_grid_config()
    
    return best_points.astype(np.int64), best_lines.astype(np.int64)

# EVOLVE-BLOCK-END
