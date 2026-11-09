# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    n = len(points)
    m = len(lines)
    # points: (n, 2), lines: (m, 3)
    # Compute A*x + B*y + C for all pairs
    # points[:, 0:1] is (n,1), lines[0:1, :] is (1, m)
    px = points[:, 0:1]  # (n, 1)
    py = points[:, 1:2]  # (n, 1)
    A = lines[0:1, :]    # (1, m)
    B = lines[1:2, :]    # (1, m)
    C = lines[2:3, :]    # (1, m)
    total = px * A + py * B + C  # (n, m)
    return np.sum(total == 0)


def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Get line through two points as (A, B, C) normalized."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x1 == x2:
        return _normalize_line(1, 0, -x1)
    elif y1 == y2:
        return _normalize_line(0, 1, -y1)
    else:
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return _normalize_line(A, B, C)


def _line_incidence_count(points, line):
    """Count how many points lie on the given line."""
    A, B, C = line
    vals = A * points[:, 0] + B * points[:, 1] + C
    return np.sum(vals == 0)


def _get_best_lines(points, num_lines=27, deadline=None):
    """Find the best num_lines lines for the given points."""
    n = len(points)
    # Generate all lines through pairs of points
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = 0
            line_counts[line] += 1
    
    # Sort by incidence count
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Greedily select lines
    selected = []
    used = set()
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line not in used:
            selected.append(line)
            used.add(line)
    
    # If we don't have enough, add zero-incidence lines
    if len(selected) < num_lines:
        all_lines = set(line_counts.keys())
        for line in all_lines:
            if line not in used:
                selected.append(line)
                used.add(line)
                if len(selected) >= num_lines:
                    break
    
    return selected[:num_lines]


def _try_grid_configuration(kx, ky, deadline):
    """Try a kx by ky grid configuration."""
    if kx * ky != 54:
        return None
    
    xs = np.arange(1, kx + 1)
    ys = np.arange(1, ky + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = _get_best_lines(points, 27, deadline)
    if lines is None or len(lines) < 27:
        return None
    
    lines_arr = np.array(lines[:27], dtype=np.int64)
    inc = _incidence_count(points, lines_arr)
    return points, lines_arr, inc


def _local_search(points, lines, max_iterations=100, deadline=None):
    """Local search to improve the configuration."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = _incidence_count(best_points, best_lines)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try moving points
        for i in range(len(best_points)):
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = best_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    
                    # Check for duplicates
                    if len(np.unique(new_points, axis=0)) != len(new_points):
                        continue
                    
                    new_lines = _get_best_lines(new_points, 27, deadline)
                    if new_lines is None or len(new_lines) < 27:
                        continue
                    
                    new_lines_arr = np.array(new_lines[:27], dtype=np.int64)
                    new_inc = _incidence_count(new_points, new_lines_arr)
                    
                    if new_inc > best_inc:
                        best_points = new_points
                        best_lines = new_lines_arr
                        best_inc = new_inc
                        improved = True
        
        # Try changing lines
        for i in range(len(best_lines)):
            # Try replacing line i with a better one
            candidate_lines = _get_best_lines(best_points, 27, deadline)
            if candidate_lines is None or len(candidate_lines) < 27:
                continue
            
            # Try swapping
            for j in range(len(candidate_lines)):
                if j == i:
                    continue
                new_lines = best_lines.copy()
                new_lines[i] = candidate_lines[j]
                
                # Check for duplicates
                norm_lines = [_normalize_line(*l) for l in new_lines]
                if len(set(norm_lines)) != len(norm_lines):
                    continue
                
                new_lines_arr = np.array(new_lines, dtype=np.int64)
                new_inc = _incidence_count(best_points, new_lines_arr)
                
                if new_inc > best_inc:
                    best_lines = new_lines_arr
                    best_inc = new_inc
                    improved = True
        
        if not improved:
            break
    
    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Try different grid configurations
    grid_configs = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    for kx, ky in grid_configs:
        if time.time() > deadline:
            break
        result = _try_grid_configuration(kx, ky, deadline)
        if result is not None:
            points, lines, inc = result
            if inc > best_inc:
                best_points = points
                best_lines = lines
                best_inc = inc
    
    # If we have a result, try local search
    if best_points is not None:
        if time.time() < deadline - 60:
            best_points, best_lines, best_inc = _local_search(
                best_points, best_lines, max_iterations=50, deadline=deadline
            )
    
    # Fallback: simple grid
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = _get_best_lines(best_points, 27, deadline)
        if best_lines is None or len(best_lines) < 27:
            # Create default lines
            best_lines = []
            for c in xs:
                best_lines.append((1, 0, -c))
            for c in ys:
                best_lines.append((0, 1, -c))
            while len(best_lines) < 27:
                best_lines.append((1, 1, 0))
        best_lines = np.array(best_lines[:27], dtype=np.int64)
    
    # Verify the result
    final_incidents = _incidence_count(best_points, best_lines)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
