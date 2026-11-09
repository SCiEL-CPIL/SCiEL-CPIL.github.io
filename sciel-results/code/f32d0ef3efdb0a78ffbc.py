# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid with all vertical, horizontal, and selected diagonal lines.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines
    lines = []
    
    # 6 Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))
        
    # 9 Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 12 Diagonal lines
    # Slope 1: y = x + c -> 1*x - 1*y + c = 0
    # Selected c values: 0, 1, 2, 3, -1, 4
    c_vals_slope_1 = [0, 1, 2, 3, -1, 4]
    for c in c_vals_slope_1:
        lines.append((1, -1, c))
        
    # Slope -1: y = -x + c -> 1*x + 1*y - c = 0
    # Selected c values: 7, 8, 9, 10, 6, 11
    c_vals_slope_neg_1 = [7, 8, 9, 10, 6, 11]
    for c in c_vals_slope_neg_1:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
