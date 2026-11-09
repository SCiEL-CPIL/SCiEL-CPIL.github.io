import numpy as np

def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Uses a 9x6 grid of points.
    Points: {(x, y) | 1 <= x <= 9, 1 <= y <= 6}
    
    Candidate lines are those aligned with the grid axes or diagonals (slopes 0, inf, 1, -1).
    We generate all such lines, calculate the number of grid points on each,
    and select the 27 lines with the highest incidence counts.
    """
    # 1. Generate Points
    W, H = 9, 6
    xs = np.arange(1, W + 1)
    ys = np.arange(1, H + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Candidate Lines
    # Each line is (A, B, C) for Ax + By + C = 0
    # We also store the count of points on the line to sort by
    candidates = []
    
    # Horizontal lines: y = y0 -> 0x + 1y - y0 = 0
    for y0 in range(1, H + 1):
        # Count points
        # All x in [1, W] are on the line
        count = W
        candidates.append(((0, 1, -y0), count))
        
    # Vertical lines: x = x0 -> 1x + 0y - x0 = 0
    for x0 in range(1, W + 1):
        # Count points
        # All y in [1, H] are on the line
        count = H
        candidates.append(((1, 0, -x0), count))
        
    # Slope 1 lines: y = x + c -> 1x - 1y + c = 0
    # c ranges such that the line intersects the grid
    # y = x + c => c = y - x
    # min c = 1 - 9 = -8
    # max c = 6 - 1 = 5
    for c in range(-8, 6):
        # Count points
        count = 0
        for x in range(1, W + 1):
            y = x + c
            if 1 <= y <= H:
                count += 1
        if count > 0:
            candidates.append(((1, -1, c), count))
            
    # Slope -1 lines: y = -x + c -> 1x + 1y - c = 0
    # c = x + y
    # min c = 1 + 1 = 2
    # max c = 9 + 6 = 15
    for c in range(2, 16):
        # Count points
        count = 0
        for x in range(1, W + 1):
            y = -x + c
            if 1 <= y <= H:
                count += 1
        if count > 0:
            candidates.append(((1, 1, -c), count))
            
    # 3. Sort and Select Top 27
    # Sort by count descending, then by line parameters for determinism
    candidates.sort(key=lambda item: (-item[1], item[0]))
    
    selected_lines = [c[0] for c in candidates[:27]]
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines
