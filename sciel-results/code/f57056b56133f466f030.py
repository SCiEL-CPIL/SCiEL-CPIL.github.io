# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    """Return a canonical integer triple for the line A*x + B*y + C = 0."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    # Make the first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidences(points, A, B, C):
    """Count how many points satisfy A*x + B*y + C == 0."""
    return int(np.count_nonzero(A * points[:, 0] + B * points[:, 1] + C == 0))


def _build_grid(m, n):
    """Build an m x n grid of points starting from (0,0) to (m-1, n-1)."""
    xs = np.arange(m, dtype=np.int64)
    ys = np.arange(n, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _get_all_lines_with_counts(points):
    """
    Enumerate all lines through at least 2 points, count incidences.
    Returns list of (count, (A,B,C)) sorted by count desc.
    """
    n = len(points)
    line_counts = {}
    
    for i in range(n):
        xi, yi = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(points[j, 0]), int(points[j, 1])
            A = yi - yj
            B = xj - xi
            C = -(xi * yj - xj * yi)
            key = _normalize(A, B, C)
            if key not in line_counts:
                line_counts[key] = 0
    
    # Count incidences for each line
    for key in list(line_counts.keys()):
        A, B, C = key
        line_counts[key] = _count_incidences(points, A, B, C)
    
    # Filter to lines with >= 2 incidences (valid non-degenerate)
    candidates = [(cnt, key) for key, cnt in line_counts.items() if cnt >= 2]
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates


def _select_top_lines(candidates, num_lines=27):
    """Select top num_lines distinct lines from sorted candidates."""
    selected = []
    seen = set()
    for cnt, key in candidates:
        if key in seen:
            continue
        seen.add(key)
        selected.append(key)
        if len(selected) == num_lines:
            break
    return selected


def _total_incidence(points, lines):
    """Compute total number of point-line incidences."""
    total = 0
    for A, B, C in lines:
        total += _count_incidences(points, A, B, C)
    return total


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Strategy: Try multiple grid configurations (different m x n shapes where m*n=54),
    for each enumerate all lines through grid points, select top 27 by incidence count,
    and return the configuration with maximum total incidences.
    """
    t_start = time.time()
    deadline = 600.0
    margin = 30.0
    
    # Factorizations of 54: 1x54, 2x27, 3x18, 6x9
    grid_shapes = [(6, 9), (3, 18), (2, 27), (9, 6), (18, 3), (27, 2)]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for m, n in grid_shapes:
        if time.time() > deadline - margin:
            break
        
        points = _build_grid(m, n)
        candidates = _get_all_lines_with_counts(points)
        
        if len(candidates) < 27:
            continue
        
        selected_lines = _select_top_lines(candidates, 27)
        score = _total_incidence(points, selected_lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = np.array(selected_lines, dtype=np.int64)
    
    # If no valid configuration found (shouldn't happen), use fallback
    if best_points is None:
        # Fallback: 6x9 grid
        best_points = _build_grid(6, 9)
        candidates = _get_all_lines_with_counts(best_points)
        selected_lines = _select_top_lines(candidates, 27)
        best_lines = np.array(selected_lines, dtype=np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
