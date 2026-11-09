import numpy as np
from math import gcd
import time
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _evaluate_points(points):
    """
    Given an array of points (N, 2), find the top 27 lines by incidence count.
    Returns (total_incidence, lines_array)
    """
    n = len(points)
    lines_dict = {}
    
    # Generate all lines from pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            key = _normalize_line(A, B, C)
            if key not in lines_dict:
                lines_dict[key] = 0

    # Count incidences for each unique line
    # Optimization: Instead of checking all points for each line, 
    # we can rely on the fact that we generated lines from pairs.
    # But we need the *total* count.
    # For N=54, number of unique lines is manageable.
    # Let's count efficiently.
    for key in lines_dict:
        A, B, C = key
        count = 0
        # Vectorized check
        # A*x + B*y + C == 0
        val = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(val == 0)
        lines_dict[key] = count

    # Sort by incidence count
    sorted_lines = sorted(lines_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 27
    if len(sorted_lines) < 27:
        # Pad with zeros if not enough lines? 
        # The problem implies we return 27 lines. 
        # If there are fewer unique lines, we might have duplicates in the selection?
        # No, "two rows describing the same line" is invalid.
        # So we must have at least 27 unique lines.
        # For 54 points in general position, there are many lines.
        # Grids have fewer, but still > 27.
        pass
        
    selected = sorted_lines[:27]
    total_inc = sum(count for _, count in selected)
    best_lines = np.array([l for l, _ in selected], dtype=np.int64)
    
    return total_inc, best_lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    best_total = -1
    best_points = None
    best_lines = None
    
    # Candidate configurations: (rows, cols, num_to_remove)
    # We will search over removals for grids slightly larger than 54.
    configs = [
        (6, 9, 0),   # Exactly 54
        (9, 6, 0),   # Exactly 54
        (7, 8, 2),   # 56 points, remove 2
        (8, 7, 2),   # 56 points, remove 2
        (5, 11, 1),  # 55 points, remove 1
        (11, 5, 1),  # 55 points, remove 1
        (4, 14, 2),  # 56 points, remove 2
        (14, 4, 2),  # 56 points, remove 2
    ]
    
    for (rows, cols, n_remove) in configs:
        if time.time() > deadline:
            break
            
        total_pts = rows * cols
        if total_pts != 54 + n_remove:
            continue
        
        # Generate grid points
        xs = np.arange(1, cols + 1)
        ys = np.arange(1, rows + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        if n_remove == 0:
            points = all_points
            total_inc, lines = _evaluate_points(points)
            if total_inc > best_total:
                best_total = total_inc
                best_points = points.copy()
                best_lines = lines
        else:
            # Search over all combinations of points to remove
            # For n_remove=1, C(55,1)=55
            # For n_remove=2, C(56,2)=1540
            # This is fast enough.
            
            # To speed up, we can precompute some things?
            # Or just loop.
            
            indices_to_remove = combinations(range(total_pts), n_remove)
            
            for combo in indices_to_remove:
                if time.time() > deadline:
                    return best_points, best_lines
                
                # Create mask of points to keep
                mask = np.ones(total_pts, dtype=bool)
                mask[list(combo)] = False
                points = all_points[mask]
                
                total_inc, lines = _evaluate_points(points)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_points = points.copy()
                    best_lines = lines
                    
    if best_points is None:
        # Fallback: 6x9 grid
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        total_inc, best_lines = _evaluate_points(best_points)
    
    return best_points, best_lines
