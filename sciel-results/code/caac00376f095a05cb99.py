import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    # 4 corners, 4 edges, 2 centers
    s_c = 1.0 / 3.0
    s_e = 1.0 / 3.0
    s_m = 1.0 / 6.0
    
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    # 4 Corners
    centers[0] = [s_c/2, s_c/2]
    sides[0] = s_c
    centers[1] = [1 - s_c/2, s_c/2]
    sides[1] = s_c
    centers[2] = [s_c/2, 1 - s_c/2]
    sides[2] = s_c
    centers[3] = [1 - s_c/2, 1 - s_c/2]
    sides[3] = s_c
    
    # 4 Edges
    centers[4] = [0.5, 1 - s_e/2] # Top
    sides[4] = s_e
    centers[5] = [0.5, s_e/2] # Bottom
    sides[5] = s_e
    centers[6] = [s_c/2, 0.5] # Left
    sides[6] = s_c # Wait, s_e is 1/3, s_c is 1/3. Same.
    sides[6] = s_e
    centers[7] = [1 - s_c/2, 0.5] # Right
    sides[7] = s_e
    
    # 2 Centers
    # Stacked vertically in the middle
    # x center is 0.5
    # y centers are 0.5 - s_m/2 and 0.5 + s_m/2
    centers[8] = [0.5, 0.5 - s_m/2]
    sides[8] = s_m
    centers[9] = [0.5, 0.5 + s_m/2]
    sides[9] = s_m
    
    return centers, sides
