# EVOLVE-BLOCK-START
import numpy as np
from math import gcd, isclose
import time


def _validate(points, lines):
    """Validate the output format. Returns True if valid."""
    if not isinstance(points, np.ndarray) or points.shape != (54, 2):
        return False
    if not isinstance(lines, np.ndarray) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.all(isclose(points, np.round(points), atol=1e-6)) or not np.all(isclose(lines, np.round(lines), atol=1e-6)):
        return False
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    # Check for repeated points
    if len(np.unique(points, axis=0)) != 54:
        return False
    # Check lines
    line_set = set()
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        # Make first nonzero entry positive
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        key = (A, B, C)
        if key in line_set:
            return False
        line_set.add(key)
    return True


def _incidence(points, lines):
    """Compute exact incidence count using int64."""
    count = 0
    for p in points:
        for l in lines:
            A, B, C = l
            if A * p[0] + B * p[1] + C == 0:
                count += 1
    return count


def _grid_construction(cols, rows, num_diag):
    """
    Construct points and lines from a cols x rows grid.
    Returns (points, lines, incidence).
    """
    # Grid points
    xs = np.arange(cols)
    ys = np.arange(rows)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.stack([px.ravel(), py.ravel()], axis=1)
    total_points = cols * rows
    
    # Vertical lines: x = i -> (1, 0, -i)
    v_lines = []
    for i in range(cols):
        v_lines.append([1, 0, -i])
    
    # Horizontal lines: y = j -> (0, 1, -j)
    h_lines = []
    for j in range(rows):
        h_lines.append([0, 1, -j])
    
    # Diagonal lines: we need num_diag diagonals
    # Diagonals of slope 1: x - y = c, i.e., (1, -1, -c) where c = x - y
    # Diagonals of slope -1: x + y = c, i.e., (1, 1, -c) where c = x + y
    # We want to pick diagonals that maximize coverage of the grid points.
    
    # Compute all possible diagonal "lines" and their coverage
    # Slope 1: x - y = c, c ranges from -(rows-1) to cols-1
    diag1 = []
    for c in range(-(rows-1), cols):
        # Count points on this diagonal
        cnt = 0
        for x in range(cols):
            y = x - c
            if 0 <= y < rows:
                cnt += 1
        diag1.append((cnt, c))
    
    # Slope -1: x + y = c, c ranges from 0 to (cols-1)+(rows-1)
    diag2 = []
    for c in range(0, cols + rows):
        cnt = 0
        for x in range(cols):
            y = c - x
            if 0 <= y < rows:
                cnt += 1
        diag2.append((cnt, c))
    
    # Pick top num_diag diagonals by coverage
    diag1.sort(reverse=True)
    diag2.sort(reverse=True)
    
    # We need to pick num_diag diagonals total. Let's try a few splits.
    best_points = None
    best_lines = None
    best_inc = -1
    
    for n1 in range(0, num_diag+1):
        n2 = num_diag - n1
        if n1 > len(diag1) or n2 > len(diag2):
            continue
        
        d1_selected = diag1[:n1]
        d2_selected = diag2[:n2]
        
        d_lines = []
        for cnt, c in d1_selected:
            d_lines.append([1, -1, -c])
        for cnt, c in d2_selected:
            d_lines.append([1, 1, -c])
        
        all_lines = v_lines + h_lines + d_lines
        if len(all_lines) != 27:
            continue
        
        # Now select 54 points from all_points to maximize incidence
        # Compute incidence contribution of each point
        # For each point, count how many lines it lies on
        point_inc = np.zeros(total_points, dtype=np.int64)
        for pt_idx in range(total_points):
            x, y = all_points[pt_idx]
            cnt = 0
            for l in all_lines:
                A, B, C = l
                if A * x + B * y + C == 0:
                    cnt += 1
            point_inc[pt_idx] = cnt
        
        # Select top 54 points by incidence
        if total_points <= 54:
            selected_idx = np.arange(total_points)
        else:
            # Use argsort to get top 54
            sorted_idx = np.argsort(point_inc)[::-1]
            selected_idx = sorted_idx[:54]
        
        points = all_points[selected_idx]
        lines = np.array(all_lines, dtype=np.int64)
        
        inc = _incidence(points, lines)
        if inc > best_inc:
            best_inc = inc
            best_points = points.copy()
            best_lines = lines.copy()
    
    if best_points is None:
        return None, None, 0
    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.
    
    Uses grid-based construction with vertical, horizontal, and diagonal lines.
    Tries multiple grid shapes and selects the best configuration.
    """
    start_time = time.time()
    deadline = 600 - 30  # Safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Try various grid shapes
    grid_shapes = [
        (3, 18), (18, 3),
        (6, 9), (9, 6),
        (7, 8), (8, 7),
        (5, 11), (11, 5),
        (4, 14), (14, 4),
        (2, 27), (27, 2),
        (3, 19), (19, 3),
        (6, 10), (10, 6),
        (9, 7), (7, 9),
        (5, 12), (12, 5),
        (4, 15), (15, 4),
    ]
    
    for cols, rows in grid_shapes:
        if time.time() > deadline:
            break
        
        if cols * rows < 54:
            continue
        
        points, lines, inc = _grid_construction(cols, rows, 27 - cols - rows)
        if points is not None and inc > best_inc:
            best_inc = inc
            best_points = points
            best_lines = lines
    
    # If no grid worked well, fall back to the original Szemeredi-Trotter construction
    if best_points is None or best_inc == 0:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
        best_inc = _incidence(best_points, best_lines)
    
    # Final validation
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
