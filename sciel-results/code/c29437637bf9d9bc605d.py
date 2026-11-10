import numpy as np
import time
from math import gcd

def count_incidences(points, lines):
    """
    Count the number of (point, line) incidences.
    points: (N, 2) array
    lines: (M, 3) array (A, B, C)
    """
    # points: (N, 2)
    # lines: (M, 3)
    # A*x + B*y + C == 0
    # Use broadcasting
    # points[:, 0] is x, points[:, 1] is y
    # lines[:, 0] is A, lines[:, 1] is B, lines[:, 2] is C
    
    # Shape: (M, 1) * (1, N) + (M, 1) * (1, N) + (M, 1) * (1, 1)
    # This creates (M, N) matrix
    x = points[:, 0].reshape(1, -1)
    y = points[:, 1].reshape(1, -1)
    A = lines[:, 0].reshape(-1, 1)
    B = lines[:, 1].reshape(-1, 1)
    C = lines[:, 2].reshape(-1, 1)
    
    vals = A * x + B * y + C
    incidences = np.sum(vals == 0)
    return incidences

def find_best_lines(points, num_lines=27):
    """
    Find the num_lines lines with the most incidences from the set of points.
    """
    n = len(points)
    line_counts = {}
    
    # Iterate over all pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure first non-zero is positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            # Count points on this line
            # We can do this efficiently by checking all points
            # But we only do this for unique lines
            if key not in line_counts:
                # Calculate count
                count = 0
                for p in points:
                    if A * p[0] + B * p[1] + C == 0:
                        count += 1
                line_counts[key] = count
            # If key exists, we already have the count
            
    # Sort lines by count
    sorted_lines = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Take top num_lines
    best_lines = [key for key, count in sorted_lines[:num_lines]]
    
    return best_lines

def generate_grid(w, h):
    xs = np.arange(w)
    ys = np.arange(h)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Define candidate point configurations
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (5, 10, 4), # 5x10 grid + 4 extra points?
    ]
    
    # Process simple grids
    for cfg in configs:
        if time.time() > deadline - 60:
            break
        if len(cfg) == 2:
            w, h = cfg
            points = generate_grid(w, h)
            if len(points) != 54:
                continue
            
            lines = find_best_lines(points, 27)
            lines_arr = np.array(lines, dtype=np.int64)
            
            score = count_incidences(points, lines_arr)
            
            if score > best_score:
                best_score = score
                best_points = points
                best_lines = lines_arr
                
    # Try a regular 27-gon configuration?
    # Points on a circle might yield different line structures.
    # But integer coordinates on a circle are sparse.
    # Pythagorean triples: (3,4,5), (5,12,13), etc.
    # Let's try a few known integer points on a circle.
    
    # If time permits, try random perturbations of the best grid?
    # Or other shapes?
    
    # For now, the grid search should be sufficient.
    
    if best_points is None:
        # Fallback to baseline
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
