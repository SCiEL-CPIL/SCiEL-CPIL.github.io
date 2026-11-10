# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy:
    1. Use a 6x9 grid of points (54 points total).
    2. Generate all unique lines determined by pairs of these points.
    3. Count the number of grid points on each line.
    4. Select the 27 lines with the highest incidence counts.
    """
    start_time = time.time()
    
    # 1. Define the 54 points on a 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate unique lines
    # A line is represented by (A, B, C) normalized such that gcd(|A|,|B|,|C|)=1
    # and the first non-zero coefficient is positive.
    lines_dict = {}
    
    n_pts = len(points)
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            
            # Equation: (y1-y2)x + (x2-x1)y + (x1y2-x2y1) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Make first non-zero positive
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines_dict:
                lines_dict[key] = 0
    
    # 3. Count incidences for each line
    # For each line, count how many points satisfy A*x + B*y + C == 0
    for key in lines_dict:
        A, B, C = key
        # Vectorized check
        # A * points[:, 0] + B * points[:, 1] + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        lines_dict[key] = count
        
    # 4. Select top 27 lines
    # Sort by incidence count descending
    sorted_lines = sorted(lines_dict.items(), key=lambda item: item[1], reverse=True)
    
    # We need exactly 27 lines. 
    # If we have fewer than 27 unique lines (unlikely with 54 points), we'd need to add more, 
    # but 54 points define at least 54+ lines (verticals, horizontals, etc.). 
    # Actually, max unique lines is O(n^2). 1431 pairs -> many duplicates. 
    # 6 verticals + 9 horizontals + others.
    # We should easily have > 27 lines with count >= 2.
    
    selected_lines = []
    for key, count in sorted_lines:
        if len(selected_lines) == 27:
            break
        selected_lines.append(key)
        
    # If for some reason we have less than 27 (should not happen), pad with dummy lines?
    # The problem guarantees we can return 27 lines. 
    # Let's ensure we have 27.
    while len(selected_lines) < 27:
        # Add a line with 0 incidences, e.g., x = 100
        selected_lines.append((1, 0, -100))
        
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
