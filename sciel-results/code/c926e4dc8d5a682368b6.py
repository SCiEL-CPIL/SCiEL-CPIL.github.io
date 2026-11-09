# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) to a canonical form."""
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _generate_best_lines(points, num_lines=27):
    """Select the best num_lines from all lines through pairs of points."""
    n_pts = len(points)
    line_map = {}     # normalized line -> (A, B, C)
    
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            if x1 == x2:
                line = (1, 0, -x1)
            elif y1 == y2:
                line = (0, 1, -y1)
            else:
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                line = (A, B, C)
            
            nline = _normalize_line(*line)
            if nline not in line_map:
                line_map[nline] = line
    
    # Count incidences for each unique line
    line_counts = {}
    for nline, (A, B, C) in line_map.items():
        count = 0
        for i in range(n_pts):
            x, y = points[i]
            if A * x + B * y + C == 0:
                count += 1
        line_counts[nline] = count
    
    # Sort by count descending, break ties by line tuple for determinism
    sorted_lines = sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Pick top num_lines
    selected = [line_map[nline] for nline, _ in sorted_lines[:num_lines]]
    return selected

def szemeredi_trotter_incidences():
    """
    Returns 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid with optimal line selection via exhaustive candidate evaluation.
    """
    # Points: 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Select best 27 lines by exhaustive evaluation
    best_lines = _generate_best_lines(points, 27)
    lines_array = np.array(best_lines, dtype=np.int64)
    
    return points, lines_array

# EVOLVE-BLOCK-END
