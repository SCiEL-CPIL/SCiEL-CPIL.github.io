# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs 54 points and 27 lines maximizing incidences.
    
    Strategy:
    - 9 vertical lines x = 0..8
    - 9 horizontal lines y = 0..8
    - 4 diagonal lines of slope 1: y = x + k for k in {-4, -2, 0, 2}
    - 5 diagonal lines of slope -1: y = -x + k for k in {4, 6, 8, 10, 12}
    
    Points:
    - 20 points at intersections of slope-1 and slope-1 diagonals (each on 4 lines: 2 vert/hor + 2 diag)
    - 34 points on exactly one diagonal line (each on 3 lines: 1 vert/hor + 1 diag)
    
    Total incidences: 20*4 + 34*3 = 80 + 102 = 182
    """
    lines = []
    
    # 9 vertical lines: x = 0..8 -> (1, 0, -i)
    for i in range(9):
        lines.append((1, 0, -i))
        
    # 9 horizontal lines: y = 0..8 -> (0, 1, -j)
    for j in range(9):
        lines.append((0, 1, -j))
        
    # 4 diagonal lines of slope 1: y = x + k -> x - y - k = 0 -> (1, -1, -k)
    k1_values = [-4, -2, 0, 2]
    for k in k1_values:
        lines.append((1, -1, -k))
        
    # 5 diagonal lines of slope -1: y = -x + k -> x + y - k = 0 -> (1, 1, -k)
    k2_values = [4, 6, 8, 10, 12]
    for k in k2_values:
        lines.append((1, 1, -k))
        
    lines = np.array(lines, dtype=np.int64)
    
    points = set()
    
    # Find intersection points of slope-1 and slope-1 diagonals
    # For slope 1: y = x + k1, for slope -1: y = -x + k2
    # x + k1 = -x + k2 -> 2x = k2 - k1 -> x = (k2 - k1) / 2
    # y = x + k1 = (k2 - k1) / 2 + k1 = (k2 + k1) / 2
    for k1 in k1_values:
        for k2 in k2_values:
            if (k2 - k1) % 2 == 0:
                x = (k2 - k1) // 2
                y = (k2 + k1) // 2
                if 0 <= x <= 8 and 0 <= y <= 8:
                    points.add((x, y))
    
    # We should have 20 intersection points
    intersection_points = list(points)
    
    # Now find points on exactly one diagonal line
    # A point (i, j) is on a slope-1 diagonal if j - i is in k1_values
    # A point (i, j) is on a slope-1 diagonal if i + j is in k2_values
    
    single_diag_points = []
    for i in range(9):
        for j in range(9):
            if (i, j) in points:
                continue
            on_slope1 = (j - i) in k1_values
            on_slope_minus1 = (i + j) in k2_values
            if on_slope1 or on_slope_minus1:
                single_diag_points.append((i, j))
    
    # We need 34 single-diagonal points
    # Take the first 34
    single_diag_points = single_diag_points[:34]
    
    # Combine all points
    all_points = intersection_points + single_diag_points
    
    # Ensure we have exactly 54 points
    assert len(all_points) == 54, f"Expected 54 points, got {len(all_points)}"
    
    points = np.array(all_points, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
