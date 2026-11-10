# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize_line(A, B, C):
    """Normalize (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    import math
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_points(p1, p2):
    """Line through two integer points (x1,y1), (x2,y2) in normalized (A,B,C) form."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _count_incidences(points, lines):
    """Count exact integer incidences."""
    n = len(points)
    m = len(lines)
    total = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                total += 1
    return total


def _generate_candidates():
    """
    Generate candidate (points, lines) configurations and return the best.
    
    Strategy: Use vertical bundles of 3 points at 18 x-locations.
    At each x, 3 points with compact y-spacing. Lines are:
      - 18 vertical lines (one per x-column)
      - 9 slanted lines (3 slopes × 3 intercepts), each hitting 3 points
    Total: 54 points, 27 lines.
    """
    best_inc = -1
    best_points = None
    best_lines = None
    
    # Vertical line parameters
    n_x = 18
    n_per_x = 3
    
    # Try different x-spacings and y-spacings
    x_spacings = [1, 2, 3, 5, 7]
    y_spacings = [1, 2, 3, 4, 5]
    x_offsets = [0, 1, 2, 3, 5]
    
    for x_step in x_spacings:
        for y_step in y_spacings:
            for x_off in x_offsets:
                # Generate points
                points_list = []
                for i in range(n_x):
                    x = x_off + i * x_step
                    for j in range(n_per_x):
                        y = 1 + j * y_step
                        points_list.append((x, y))
                
                points = np.array(points_list, dtype=np.int64)
                
                # Check for duplicate points
                if len(set(map(tuple, points.tolist()))) != len(points):
                    continue
                
                # Generate lines
                lines_set = set()
                
                # Vertical lines: x = c  ->  (1, 0, -c)
                for i in range(n_x):
                    x = x_off + i * x_step
                    lines_set.add(_normalize_line(1, 0, -x))
                
                # Slanted lines: try slopes 1, 2, 3 with various intercepts
                # For each slope s, find lines y = s*x + b that pass through 3 points
                for s in [1, 2, 3]:
                    # For each point, compute b = y - s*x, group by b
                    b_counts = {}
                    for idx, (x, y) in enumerate(points_list):
                        b = y - s * x
                        if b not in b_counts:
                            b_counts[b] = []
                        b_counts[b].append(idx)
                    
                    # Pick top b values by count
                    sorted_bs = sorted(b_counts.keys(), key=lambda b: len(b_counts[b]), reverse=True)
                    for b in sorted_bs[:3]:
                        if len(b_counts[b]) >= 2:
                            # Line: s*x - y + b = 0  ->  (s, -1, b)
                            norm = _normalize_line(s, -1, b)
                            lines_set.add(norm)
                
                # If we don't have 27 lines yet, add more
                if len(lines_set) < 27:
                    # Add more slanted lines with other slopes
                    for s in [4, 5, -1, -2]:
                        b_counts = {}
                        for idx, (x, y) in enumerate(points_list):
                            b = y - s * x
                            if b not in b_counts:
                                b_counts[b] = []
                            b_counts[b].append(idx)
                        sorted_bs = sorted(b_counts.keys(), key=lambda b: len(b_counts[b]), reverse=True)
                        for b in sorted_bs:
                            if len(lines_set) >= 27:
                                break
                            if len(b_counts[b]) >= 2:
                                norm = _normalize_line(s, -1, b)
                                lines_set.add(norm)
                
                if len(lines_set) != 27:
                    continue
                
                lines = np.array(list(lines_set), dtype=np.int64)
                
                # Verify no duplicate lines
                if len(set(map(tuple, lines.tolist()))) != 27:
                    continue
                
                # Count incidences
                inc = _count_incidences(points, lines)
                
                if inc > best_inc:
                    best_inc = inc
                    best_points = points.copy()
                    best_lines = lines.copy()
    
    # Fallback: use the known good construction if search didn't find anything
    if best_points is None:
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
    
    return best_points, best_lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization: 54 points, 27 lines.
    Returns (points, lines) maximizing incidences.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    # Try the structured search
    try:
        points, lines = _generate_candidates()
    except Exception:
        # Fallback to baseline
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
    
    # Ensure correct shapes and types
    points = np.asarray(points, dtype=np.int64)
    lines = np.asarray(lines, dtype=np.int64)
    
    # Final validation
    assert points.shape == (54, 2), f"Points shape {points.shape}"
    assert lines.shape == (27, 3), f"Lines shape {lines.shape}"
    
    # Check no NaN/inf (should be fine with int64)
    # Check no repeated points
    pt_tuples = set(map(tuple, points.tolist()))
    assert len(pt_tuples) == 54, "Duplicate points"
    
    # Check no line with A=B=0
    for row in lines:
        assert not (row[0] == 0 and row[1] == 0), "Line with A=B=0"
    
    # Check no duplicate lines
    ln_tuples = set()
    for row in lines:
        A, B, C = int(row[0]), int(row[1]), int(row[2])
        norm = _normalize_line(A, B, C)
        assert norm not in ln_tuples, "Duplicate lines"
        ln_tuples.add(norm)
    
    return points, lines


# EVOLVE-BLOCK-END
