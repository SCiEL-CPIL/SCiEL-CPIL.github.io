import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def szemeredi_trotter_incidences():
    """
    Returns a configuration of 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid with optimal diagonal lines.
    """
    # 1. Define Points: 6x9 grid
    # x in 0..5, y in 0..8
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
            
    # 2. Define Lines
    lines = []
    
    # 6 Vertical lines: x = 0..5
    for i in range(6):
        lines.append(_normalize_line(1, 0, -i))
        
    # 9 Horizontal lines: y = 0..8
    for j in range(9):
        lines.append(_normalize_line(0, 1, -j))
        
    # 6 Slope 1 lines: x + y = c
    # Optimal c values for 6x9 grid (x:0-5, y:0-8)
    # Counts: c=4,5,6,7,8,9 have 5,6,6,6,6,5 points respectively.
    # We pick the 6 with highest counts.
    optimal_c_slope1 = [4, 5, 6, 7, 8, 9]
    for c in optimal_c_slope1:
        lines.append(_normalize_line(1, 1, -c))
        
    # 6 Slope -1 lines: x - y = c
    # Optimal c values for 6x9 grid
    # Counts: c=-4,-3,-2,-1,0,1 have 5,6,6,6,6,5 points respectively.
    optimal_c_slope_minus1 = [-4, -3, -2, -1, 0, 1]
    for c in optimal_c_slope_minus1:
        lines.append(_normalize_line(1, -1, -c))
        
    # Convert to arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points_arr, lines_arr
