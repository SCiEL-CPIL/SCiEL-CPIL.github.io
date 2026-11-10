import numpy as np
import time
from math import gcd

def normalize_line(A, B, C):
    """
    Normalize the line coefficients (A, B, C) such that:
    1. gcd(|A|, |B|, |C|) == 1
    2. The first non-zero coefficient is positive.
    """
    # Find gcd of absolute values
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        # Degenerate case, shouldn't happen for valid lines
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    
    # Ensure first non-zero is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def get_line_from_points(p1, p2):
    """
    Given two points p1, p2 (tuples of ints), return the normalized (A, B, C)
    for the line passing through them.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Normal vector (A, B)
    # A*x + B*y + C = 0
    # A = dy, B = -dx
    A = dy
    B = -dx
    C = -(A * x1 + B * y1)
    
    return normalize_line(A, B, C)

def count_incidences(points, line):
    """
    Count how many points in 'points' lie on 'line' (A, B, C).
    """
    A, B, C = line
    count = 0
    for x, y in points:
        if A * x + B * y + C == 0:
            count += 1
    return count

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Define the 6x9 grid
    # Points: {1, ..., 6} x {1, ..., 9}
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_list = list(zip(px.ravel(), py.ravel()))
    
    # 2. Generate all unique lines determined by pairs of points
    unique_lines = set()
    n_points = len(points_list)
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            line = get_line_from_points(points_list[i], points_list[j])
            unique_lines.add(line)
            
    # 3. Calculate incidence count for each unique line
    line_scores = []
    for line in unique_lines:
        score = count_incidences(points_list, line)
        line_scores.append((score, line))
        
    # 4. Sort lines by score descending, then by line coeffs for determinism
    line_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # 5. Select top 27 lines
    top_lines = [ls[1] for ls in line_scores[:27]]
    
    # 6. Prepare output arrays
    points_arr = np.array(points_list, dtype=np.int64)
    lines_arr = np.array(top_lines, dtype=np.int64)
    
    return points_arr, lines_arr
