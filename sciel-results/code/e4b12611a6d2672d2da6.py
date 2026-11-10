# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences exactly."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _normalize_line(A, B, C):
    """Normalize a line so it's canonical."""
    from math import gcd
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _is_valid(points, lines):
    """Check validity constraints."""
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    # Check no NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    # Check integer-ness
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Check repeated points
    pts = [tuple(p) for p in points]
    if len(set(pts)) != 54:
        return False
    # Check lines
    seen_lines = set()
    for A, B, C in lines:
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _make_grid_construction():
    """6x9 grid with vertical, horizontal, and diagonal lines."""
    # Points: 6 x 9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    # 6 vertical lines: x = i -> (1, 0, -i)
    for i in range(6):
        lines.append((1, 0, -i))
    # 9 horizontal lines: y = j -> (0, 1, -j)
    for j in range(9):
        lines.append((0, 1, -j))
    # 12 diagonal lines with slope 1: y - x = c
    # Diagonals c from -5 to 8, pick longest 12
    diag_lengths = {}
    for c in range(-5, 9):
        # Count points where y - x = c
        cnt = 0
        for x in range(6):
            y = x + c
            if 0 <= y < 9:
                cnt += 1
        diag_lengths[c] = cnt
    # Sort by length descending
    sorted_diags = sorted(diag_lengths.items(), key=lambda x: -x[1])[:12]
    for c, _ in sorted_diags:
        lines.append((1, -1, c))  # x - y + c = 0 -> y = x + c
    
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def _local_search(points, lines, max_time=300):
    """Local search to improve incidence count."""
    start_time = time.time()
    current_inc = _incidence_count(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = current_inc
    
    # Try swapping lines: replace one line with a better one
    # Candidate lines: all lines through pairs of points
    def get_candidate_lines(points):
        """Generate all distinct lines through pairs of points."""
        n = len(points)
        line_set = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                norm = _normalize_line(int(A), int(B), int(C))
                if norm not in line_set:
                    line_set[norm] = (norm, 0)  # (line, count)
                line_set[norm] = (norm, line_set[norm][1] + 1)
        return list(line_set.keys())
    
    # Try moving individual points
    def try_move_point(points, lines, p_idx, new_pt, current_inc):
        """Compute new incidence count if we move point p_idx to new_pt."""
        old_pt = points[p_idx]
        old_x, old_y = old_pt
        new_x, new_y = new_pt
        
        # Count incidences lost from old point
        lost = 0
        for A, B, C in lines:
            if A * old_x + B * old_y + C == 0:
                lost += 1
        
        # Count incidences gained at new point
        gained = 0
        for A, B, C in lines:
            if A * new_x + B * new_y + C == 0:
                gained += 1
        
        return current_inc - lost + gained
    
    # Try replacing a line
    def try_replace_line(points, lines, l_idx, new_line, current_inc):
        """Compute new incidence count if we replace line l_idx with new_line."""
        old_line = lines[l_idx]
        old_A, old_B, old_C = old_line
        new_A, new_B, new_C = new_line
        
        # Count incidences lost from old line
        lost = 0
        for x, y in points:
            if old_A * x + old_B * y + old_C == 0:
                lost += 1
        
        # Count incidences gained by new line
        gained = 0
        for x, y in points:
            if new_A * x + new_B * y + new_C == 0:
                gained += 1
        
        return current_inc - lost + gained
    
    # Precompute candidate lines
    candidate_lines = get_candidate_lines(points)
    
    iterations = 0
    while time.time() - start_time < max_time:
        iterations += 1
        improved = False
        
        # Try moving points to nearby lattice positions
        for p_idx in range(54):
            if time.time() - start_time > max_time:
                break
            x, y = points[p_idx]
            # Try nearby positions
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if new_x < -10 or new_x > 20 or new_y < -10 or new_y > 20:
                        continue
                    # Check no duplicate points
                    dup = False
                    for q_idx in range(54):
                        if q_idx != p_idx and points[q_idx][0] == new_x and points[q_idx][1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    new_inc = try_move_point(points, lines, p_idx, (new_x, new_y), current_inc)
                    if new_inc > current_inc:
                        points[p_idx] = [new_x, new_y]
                        current_inc = new_inc
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
            # Try replacing lines
            for l_idx in range(27):
                if time.time() - start_time > max_time:
                    break
                best_gain = 0
                best_line = None
                for cand in candidate_lines:
                    # Skip if it's the same as current line
                    if _normalize_line(int(lines[l_idx][0]), int(lines[l_idx][1]), int(lines[l_idx][2])) == cand:
                        continue
                    new_inc = try_replace_line(points, lines, l_idx, cand, current_inc)
                    gain = new_inc - current_inc
                    if gain > best_gain:
                        best_gain = gain
                        best_line = cand
                if best_line is not None and best_gain > 0:
                    lines[l_idx] = best_line
                    current_inc += best_gain
                    improved = True
                    if current_inc > best_inc:
                        best_inc = current_inc
                        best_points = points.copy()
                        best_lines = lines.copy()
                    # Update candidate lines
                    candidate_lines = get_candidate_lines(points)
                    break
            if not improved:
                break
    
    return best_points, best_lines, best_inc

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    
    # Start with grid construction
    points, lines = _make_grid_construction()
    
    # Local search
    points, lines, inc = _local_search(points, lines, max_time=500)
    
    # Final validity check
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    if not _is_valid(points, lines):
        # Fallback to simple grid
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

# EVOLVE-BLOCK-END
