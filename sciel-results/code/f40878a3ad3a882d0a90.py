# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from collections import defaultdict


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so that:
    - gcd(|A|, |B|, |C|) divides all coefficients (divide by gcd)
    - first nonzero entry is positive
    Returns tuple (a, b, c) with small ints.
    """
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count exact incidences using int64 arithmetic."""
    count = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count


def _find_best_lines_for_points(points, num_lines=27):
    """Find the num_lines lines that maximize incidences with the given points.
    
    Uses a dual-space approach: enumerate all lines through pairs of points,
    count how many points each line passes through, and select the top num_lines.
    """
    n = len(points)
    line_counts = defaultdict(int)
    
    # For each pair of points, compute the line through them
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Line: dy*(x - x1) - dx*(y - y1) = 0
            # dy*x - dx*y + (dx*y1 - dy*x1) = 0
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts[norm] += 1
    
    # Also add horizontal and vertical lines (they might not be captured by pairs if all points are distinct)
    # Actually, pairs already capture all lines through at least 2 points.
    # But we should also consider lines through only 1 point? No, those have count 1.
    # The top 27 will almost certainly have count >= 2 for reasonable point sets.
    
    # Get top num_lines
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    top = sorted_lines[:num_lines]
    
    if len(top) < num_lines:
        # Pad with dummy lines (shouldn't happen for 54 points)
        while len(top) < num_lines:
            top.append(((0, 1, 0), 0))
    
    total_inc = sum(cnt for _, cnt in top)
    lines_arr = np.array([norm for norm, _ in top], dtype=np.int64)
    
    return lines_arr, total_inc


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Approach: Try structured point sets (grids with various aspect ratios) and
    for each, find the optimal 27 lines. The key insight is that rectangular grids
    have many collinearities along rows, columns, and diagonals.
    
    For an X x Y grid (X*Y=54), the number of incidences from:
    - X vertical lines: each has Y points -> X*Y = 54
    - Y horizontal lines: each has X points -> Y*X = 54
    - Diagonal lines: various counts depending on gcd(X,Y) and offsets
    
    We evaluate several grid configurations and return the best.
    """
    
    best_inc = -1
    best_points = None
    best_lines = None
    
    # Try all factor pairs of 54
    factors = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    
    for X, Y in factors:
        # Generate X x Y grid points
        xs = np.arange(1, X + 1, dtype=np.int64)
        ys = np.arange(1, Y + 1, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        cand_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Find best 27 lines
        lines_arr, inc = _find_best_lines_for_points(cand_points, 27)
        
        if inc > best_inc:
            best_inc = inc
            best_points = cand_points.copy()
            best_lines = lines_arr.copy()
    
    # Also try some non-grid structured sets
    
    # Strategy: Points on 3 parallel lines with 18 points each, spaced to create diagonals
    # Line 1: y=0, x=1..18
    # Line 2: y=10, x=1..18  
    # Line 3: y=20, x=1..18
    # This creates verticals (18 each), horizontals (18 each), and diagonals
    
    for y_offset in [1, 2, 3, 5, 10, 15, 20]:
        points_cand = np.zeros((54, 2), dtype=np.int64)
        for line_idx in range(3):
            for i in range(18):
                points_cand[line_idx * 18 + i] = [i + 1, line_idx * y_offset]
        
        lines_arr, inc = _find_best_lines_for_points(points_cand, 27)
        if inc > best_inc:
            best_inc = inc
            best_points = points_cand.copy()
            best_lines = lines_arr.copy()
    
    # Strategy: Points on 2 parallel lines with 27 points each
    for y_offset in [1, 2, 3, 5, 10]:
        points_cand = np.zeros((54, 2), dtype=np.int64)
        for i in range(27):
            points_cand[i] = [i + 1, 0]
            points_cand[i + 27] = [i + 1, y_offset]
        
        lines_arr, inc = _find_best_lines_for_points(points_cand, 27)
        if inc > best_inc:
            best_inc = inc
            best_points = points_cand.copy()
            best_lines = lines_arr.copy()
    
    # Strategy: Points on a single line (all 54 collinear)
    points_cand = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        points_cand[i] = [i + 1, 0]
    lines_arr, inc = _find_best_lines_for_points(points_cand, 27)
    if inc > best_inc:
        best_inc = inc
        best_points = points_cand.copy()
        best_lines = lines_arr.copy()
    
    # Strategy: 6x9 grid with shifted rows to create more diagonal incidences
    # Row 0: x=1..6, y=1
    # Row 1: x=1..6, y=2
    # ... but with some rows shifted
    
    # Try: 6 columns, 9 rows, but alternate rows shifted by 1
    for shift_pattern in [0, 1]:
        points_cand = np.zeros((54, 2), dtype=np.int64)
        for row in range(9):
            shift = shift_pattern * (row % 2)
            for col in range(6):
                idx = row * 6 + col
                points_cand[idx] = [col + 1 + shift, row + 1]
        
        lines_arr, inc = _find_best_lines_for_points(points_cand, 27)
        if inc > best_inc:
            best_inc = inc
            best_points = points_cand.copy()
            best_lines = lines_arr.copy()
    
    # Ensure we have a valid result
    if best_points is None or best_lines is None:
        # Fallback: 3x18 grid
        X, Y = 3, 18
        xs = np.arange(1, X + 1, dtype=np.int64)
        ys = np.arange(1, Y + 1, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines, _ = _find_best_lines_for_points(best_points, 27)
    
    # Validate shapes
    assert best_points.shape == (54, 2), f"Points shape {best_points.shape}"
    assert best_lines.shape == (27, 3), f"Lines shape {best_lines.shape}"
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
