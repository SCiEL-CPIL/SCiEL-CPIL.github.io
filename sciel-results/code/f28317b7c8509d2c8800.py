# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _normalize_line(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True


def _count_incidences(points, lines):
    # points: (54,2) int array, lines: (27,3) int array
    # Compute A*x + B*y + C for all pairs
    # points[:, 0:1] -> (54,1), points[:, 1:2] -> (54,1)
    # lines[:, 0:1] -> (27,1), lines[:, 1:2] -> (27,1), lines[:, 2:3] -> (27,1)
    x = points[:, 0:1].astype(np.int64)  # (54,1)
    y = points[:, 1:2].astype(np.int64)  # (54,1)
    A = lines[:, 0:1].astype(np.int64)   # (27,1)
    B = lines[:, 1:2].astype(np.int64)   # (27,1)
    C = lines[:, 2:3].astype(np.int64)   # (27,1)
    # (54,1) * (1,27) broadcast -> (54,27)
    vals = x * A.T + y * B.T + C.T
    return int(np.sum(vals == 0))


def _best_lines_for_points(points):
    """Given 54 points, find the 27 lines with max total incidences (no duplicates)."""
    n = len(points)
    # Generate all lines through pairs of points
    line_counts = {}  # normalized (A,B,C) -> count
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(int(A), int(B), int(C))
            if key[0] == 0 and key[1] == 0:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Also consider lines that might be "better" even if not through a pair?
    # In integer lattice with integer points, any line through 2+ points is captured.
    # Lines through only 1 point are suboptimal unless we have <27 multi-point lines.
    
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
    
    # Pick top 27
    selected = [key for key, cnt in sorted_lines[:27]]
    
    # If we have fewer than 27, pad with single-point lines (rare)
    if len(selected) < 27:
        # Add arbitrary lines through single points
        existing_set = set(selected)
        for i in range(n):
            if len(selected) >= 27:
                break
            x, y = points[i]
            # Try a few simple lines through this point
            for A, B, C in [(1, 0, -x), (0, 1, -y), (1, 1, -(x+y)), (1, -1, y-x)]:
                key = _normalize_line(A, B, C)
                if key not in existing_set and key[0] != 0 or key[1] != 0:
                    if not (key[0] == 0 and key[1] == 0):
                        selected.append(key)
                        existing_set.add(key)
                        break
    
    return np.array(selected[:27], dtype=np.int64)


def _initial_grid():
    """6x9 grid."""
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = [(x, y) for x in xs for y in ys]
    points = np.array(points, dtype=np.int64)
    lines = _best_lines_for_points(points)
    return points, lines


def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    
    # Initialize
    points, lines = _initial_grid()
    best_points = points.copy()
    best_lines = lines.copy()
    best_count = _count_incidences(points, lines)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute some random offsets for speed
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - safety_margin and T > T_min:
        # Perturb points: move a random point by a small random integer offset
        idx = rng.integers(0, 54)
        dx = rng.integers(-2, 3)
        dy = rng.integers(-2, 3)
        if dx == 0 and dy == 0:
            continue
        
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        
        # Check validity of new points (no duplicates, finite, integer)
        valid = True
        pt_set = set()
        for i in range(54):
            x, y = int(new_points[i, 0]), int(new_points[i, 1])
            if (x, y) in pt_set:
                valid = False
                break
            pt_set.add((x, y))
        if not valid:
            T *= cooling
            continue
        
        # Recompute best lines for new points
        new_lines = _best_lines_for_points(new_points)
        
        # Check validity of lines
        if not _is_valid(new_points, new_lines):
            T *= cooling
            continue
        
        new_count = _count_incidences(new_points, new_lines)
        
        # Acceptance criterion
        if new_count > best_count:
            points = new_points
            lines = new_lines
            best_points = points.copy()
            best_lines = lines.copy()
            best_count = new_count
            T = max(T * 0.99, T_min)  # Reheat slightly on improvement
        else:
            # Accept with probability exp((new_count - best_count) / T)
            delta = new_count - best_count
            if delta < 0 and rng.random() < np.exp(delta / T):
                points = new_points
                lines = new_lines
                # Don't update best
            T *= cooling
    
    # Final polish: try a few more iterations with smaller T
    T = 0.5
    for _ in range(500):
        if time.time() > deadline - safety_margin:
            break
        idx = rng.integers(0, 54)
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)
        if dx == 0 and dy == 0:
            continue
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        
        valid = True
        pt_set = set()
        for i in range(54):
            x, y = int(new_points[i, 0]), int(new_points[i, 1])
            if (x, y) in pt_set:
                valid = False
                break
            pt_set.add((x, y))
        if not valid:
            continue
        
        new_lines = _best_lines_for_points(new_points)
        if not _is_valid(new_points, new_lines):
            continue
        
        new_count = _count_incidences(new_points, new_lines)
        if new_count > best_count:
            best_points = new_points.copy()
            best_lines = new_lines.copy()
            best_count = new_count
            points = new_points
            lines = new_lines
    
    return best_points, best_lines
# EVOLVE-BLOCK-END
