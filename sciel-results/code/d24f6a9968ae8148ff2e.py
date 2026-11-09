import numpy as np
from math import gcd
from functools import reduce
import time
from itertools import combinations


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidence(points, A, B, C):
    """Count points on line A*x+B*y+C=0."""
    return int(np.sum(A * points[:, 0] + B * points[:, 1] + C == 0))


def _build_grid(a, b):
    """Build a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _generate_all_candidate_lines(points):
    """Generate a comprehensive set of candidate lines through point pairs and standard families."""
    n = len(points)
    seen = set()
    unique_lines = []
    
    def add_line(A, B, C):
        norm = _normalize(A, B, C)
        if norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
    
    # Lines through every pair of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = int(y2 - y1)
            B = int(x1 - x2)
            C = int(-(A * x1 + B * y1))
            add_line(A, B, C)
    
    # Standard families for completeness
    max_coord = max(points.max(), 20)
    
    # Horizontal: x = k
    for k in range(1, max_coord + 1):
        add_line(1, 0, -k)
    
    # Vertical: y = k
    for k in range(1, max_coord + 1):
        add_line(0, 1, -k)
    
    # Various slopes
    for slope_num, slope_den in [(1,1), (-1,1), (2,1), (1,2), (-2,1), (1,-2), 
                                   (3,1), (1,3), (-3,1), (1,-3), (3,2), (2,3),
                                   (-3,2), (2,-3), (4,1), (1,4), (-4,1), (1,-4)]:
        # slope = slope_num/slope_den: y = (slope_num/slope_den)*x + c
        # slope_num*x - slope_den*y + slope_den*c = 0
        # A = slope_num, B = -slope_den, C = slope_den*c
        A_base = slope_num
        B_base = -slope_den
        # Range of C values to cover the grid
        range_c = int(max_coord * max(abs(slope_num), abs(slope_den))) + 10
        for c in range(-range_c, range_c + 1):
            add_line(A_base, B_base, slope_den * c)
    
    return unique_lines


def _greedy_select_lines(points, candidate_lines, num_lines):
    """Greedy selection: at each step, pick the line that maximizes NEW incidences."""
    n = len(points)
    points_arr = points  # (n, 2)
    
    # Precompute for each candidate line which points it covers
    line_masks = []
    for (A, B, C) in candidate_lines:
        mask = (A * points_arr[:, 0] + B * points_arr[:, 1] + C == 0)
        line_masks.append(mask)
    
    selected = []
    covered = np.zeros(n, dtype=np.int64)
    total = 0
    
    for _ in range(num_lines):
        best_idx = -1
        best_new = -1
        for i, mask in enumerate(line_masks):
            if i in selected:
                continue
            new_count = int(np.sum(mask & (covered == 0)))
            if new_count > best_new:
                best_new = new_count
                best_idx = i
        if best_idx == -1 or best_new <= 0:
            break
        selected.append(best_idx)
        covered[line_masks[best_idx]] += 1
        total += best_new
    
    return [candidate_lines[i] for i in selected], total


def _evaluate(points, lines):
    """Compute total incidences."""
    total = 0
    for (A, B, C) in lines:
        total += _count_incidence(points, A, B, C)
    return total


def _try_grid_config(a, b, num_lines=27, deadline=None):
    """Try an a x b grid with optimal greedy line selection."""
    if a * b != 54:
        return -1, None, None
    
    points = _build_grid(a, b)
    candidate_lines = _generate_all_candidate_lines(points)
    
    if deadline and time.time() > deadline - 30:
        return -1, None, None
    
    selected, total = _greedy_select_lines(points, candidate_lines, num_lines)
    
    if len(selected) < num_lines:
        return -1, None, None
    
    # Verify no duplicate lines
    norms = set(selected)
    if len(norms) != len(selected):
        return -1, None, None
    
    lines_arr = np.array(selected, dtype=np.int64)
    return total, points, lines_arr


def _try_nonuniform_grid(a, b, num_lines=27, deadline=None):
    """Try a non-uniform grid where we can adjust spacing."""
    if a * b != 54:
        return -1, None, None
    
    # Use uniform grid as base (non-uniform is complex, skip for now)
    return _try_grid_config(a, b, num_lines, deadline)


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_total = -1
    best_points = None
    best_lines = None
    
    # Try various grid configurations - balanced grids tend to work better
    grid_configs = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)]
    
    for a, b in grid_configs:
        if time.time() > deadline - 60:
            break
        total, pts, lns = _try_grid_config(a, b, 27, deadline)
        if total > best_total:
            best_total = total
            best_points = pts
            best_lines = lns
    
    if best_points is None:
        # Fallback: 6x9 grid with basic lines
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        c_vals = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
        for c in c_vals:
            lines.append((1, -1, c))
        best_lines = np.array(lines, dtype=np.int64)
        best_total = _evaluate(best_points, best_lines)
    
    return best_points, best_lines
