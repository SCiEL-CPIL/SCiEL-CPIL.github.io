# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def szemeredi_trotter_incidences():
    """
    Maximizes incidences between 54 points and 27 lines.
    Uses a 6x9 grid and greedily selects the 27 lines with the highest incidence counts.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    # 1. Define the 54 points on a 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate all unique lines passing through at least 2 points
    lines_set = set()
    n_points = len(points)
    
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            A = -dy
            B = dx
            C = -(A * x1 + B * y1)
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            lines_set.add((A, B, C))
            
            if time.time() > deadline - safety_margin:
                break
        if time.time() > deadline - safety_margin:
            break

    # 3. Calculate incidence count for each line
    lines_list = list(lines_set)
    incidences = []
    
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(val == 0)
        if count > 0:
            incidences.append((count, (A, B, C)))
        
        if time.time() > deadline - safety_margin:
            break

    # 4. Sort by incidence count descending
    incidences.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Select top 27 lines
    selected_lines = []
    for count, line in incidences:
        if len(selected_lines) >= 27:
            break
        selected_lines.append(line)
        
    if len(selected_lines) < 27:
        # Fallback if somehow not enough lines (should not happen)
        # Generate a default set
        pass

    lines_array = np.array(selected_lines, dtype=np.int64)
    
    return points, lines_array

# EVOLVE-BLOCK-END
