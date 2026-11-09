# EVOLVE-BLOCK-START
import numpy as np
from math import gcd

def szemeredi_trotter_incidences():
    # Define the 54 points as a 6x9 grid
    # x in {0, ..., 5}, y in {0, ..., 8}
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    n_points = len(points)
    
    # Helper to normalize line (A, B, C)
    def normalize(A, B, C):
        # Find gcd of absolute values
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g == 0:
            return (0, 0, 0)
        A //= g
        B //= g
        C //= g
        # Make first non-zero positive
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    # Collect unique lines defined by pairs of points
    unique_lines = set()
    
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            key = normalize(A, B, C)
            unique_lines.add(key)
            
    # Count incidences for each unique line
    line_incidence = {}
    for line in unique_lines:
        A, B, C = line
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        line_incidence[line] = count
        
    # Sort lines by incidence count descending
    sorted_lines = sorted(line_incidence.items(), key=lambda item: item[1], reverse=True)
    
    # Select top 27 lines
    top_27_lines = [line for line, count in sorted_lines[:27]]
    
    # Convert to numpy arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(top_27_lines, dtype=np.int64)
    
    return points_arr, lines_arr
# EVOLVE-BLOCK-END
