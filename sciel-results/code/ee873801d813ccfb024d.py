# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from itertools import combinations


def _canonical_line(a, b, c):
    """Normalize line (A,B,C) so first nonzero entry is positive and gcd divides all."""
    if a == 0 and b == 0:
        return (0, 0, 0)
    g = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
    if g > 0:
        a, b, c = a // g, b // g, c // g
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)


def _line_from_points(p1, p2):
    """Given two points (x1,y1) and (x2,y2), return canonical line (A,B,C)."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return _canonical_line(a, b, c)


def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where A*x+B*y+C==0."""
    total = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total


def _get_top_k_lines(points, k=27):
    """Find the top k lines by incidence count for the given points."""
    n = len(points)
    line_counts = {}
    
    # Generate all lines from pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = 0
            line_counts[line] += 1
            
    # The count in line_counts is the number of pairs, not points.
    # We need to count actual points on each line.
    # For a line defined by (A,B,C), count points (x,y) such that Ax+By+C=0.
    
    # To be efficient, we can just iterate through unique lines and count points.
    # There are at most n(n-1)/2 lines. For n=54, this is ~1400.
    # Counting points for each line is O(n). Total O(n^3) ~ 150k ops. Very fast.
    
    unique_lines = list(line_counts.keys())
    scored_lines = []
    for line in unique_lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        scored_lines.append((count, line))
        
    # Sort by count descending
    scored_lines.sort(key=lambda x: -x[0])
    
    # Pick top k
    top_lines = [l[1] for l in scored_lines[:k]]
    
    return top_lines


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences with 54 points and 27 lines.
    Uses a 6x9 grid which is known to be optimal or near-optimal for this size.
    """
    # Create 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Find top 27 lines
    lines_list = _get_top_k_lines(points, 27)
    
    lines_arr = np.array(lines_list, dtype=np.int64).reshape(27, 3)
    
    # Validation
    if points.shape != (54, 2):
        raise ValueError("Points shape mismatch")
    if lines_arr.shape != (27, 3):
        raise ValueError("Lines shape mismatch")
        
    # Check for duplicate lines
    canonical_lines = [_canonical_line(*l) for l in lines_arr]
    if len(set(canonical_lines)) != 27:
        # Fallback: if duplicates exist, try to fix, though unlikely for grid
        unique = []
        seen = set()
        for cl in canonical_lines:
            if cl not in seen:
                seen.add(cl)
                unique.append(cl)
        if len(unique) < 27:
             # This shouldn't happen, but if it does, we might need to add more lines
             # For now, just return what we have, hoping it's valid
             pass
        lines_arr = np.array(unique[:27], dtype=np.int64).reshape(27, 3)
    
    # Check for duplicate points
    point_tuples = [tuple(p) for p in points]
    if len(set(point_tuples)) != 54:
        raise ValueError("Duplicate points found")
        
    return points, lines_arr

# EVOLVE-BLOCK-END
