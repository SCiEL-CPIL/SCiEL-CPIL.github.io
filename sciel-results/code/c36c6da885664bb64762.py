"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Grid-based cellular decomposition: 3x7 grid in a 0.6 x 1.4 rectangle
    # Perimeter = 2*(0.6+1.4) = 4.0
    # Each cell is 0.2 x 0.2, inradius = 0.1
    # 21 circles, sum of radii = 21 * 0.1 = 2.1
    
    m = 3  # rows
    n = 7  # cols
    w = 0.6
    h = 1.4
    
    cell_w = w / m
    cell_h = h / n
    r = min(cell_w, cell_h) / 2.0
    
    circles = []
    for i in range(m):
        for j in range(n):
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            circles.append([cx, cy, r])
    
    return np.array(circles[:N])
# EVOLVE-BLOCK-END
