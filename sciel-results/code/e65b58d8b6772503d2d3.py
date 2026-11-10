# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a 7x8 grid with 2 points removed to balance line lengths and coverage.
    """
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # 1. Generate 7x8 grid
    W, H = 7, 8
    xs = np.arange(1, W + 1)
    ys = np.arange(1, H + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_full = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Define candidate lines
    lines = []
    # Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, W + 1):
        lines.append((1, 0, -i))
        
    # Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, H + 1):
        lines.append((0, 1, -j))
        
    # Diagonal lines (Slope 1): y - x = c -> -1*x + 1*y - c = 0
    # Calculate length for each c
    diag1_lengths = []
    for c in range(1 - W, H): # y-x range
        # x in [1, W], y in [1, H]
        # y = x + c => 1 <= x+c <= H => 1-c <= x <= H-c
        x_min = max(1, 1 - c)
        x_max = min(W, H - c)
        if x_min <= x_max:
            length = x_max - x_min + 1
            diag1_lengths.append((length, c))
            
    # Diagonal lines (Slope -1): y + x = c -> 1*x + 1*y - c = 0
    diag2_lengths = []
    for c in range(1 + 1, W + H + 1): # y+x range
        # y = c - x => 1 <= c-x <= H => c-H <= x <= c-1
        x_min = max(1, c - H)
        x_max = min(W, c - 1)
        if x_min <= x_max:
            length = x_max - x_min + 1
            diag2_lengths.append((length, c))
            
    # Sort diagonals by length descending
    diag1_lengths.sort(reverse=True)
    diag2_lengths.sort(reverse=True)
    
    # We need 27 lines total.
    # We have W + H = 15 H/V lines.
    # Need 12 Diagonals.
    # Merge and pick top 12
    all_diags = []
    for length, c in diag1_lengths:
        all_diags.append((length, -1, 1, -c)) # A=-1, B=1, C=-c
    for length, c in diag2_lengths:
        all_diags.append((length, 1, 1, -c))  # A=1, B=1, C=-c
        
    all_diags.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 12
    selected_diags = all_diags[:12]
    
    for _, A, B, C in selected_diags:
        lines.append((A, B, C))
        
    lines = np.array(lines, dtype=np.int64)
    
    # 3. Calculate incidences for each point in the full grid
    # points_full shape (56, 2)
    # lines shape (27, 3)
    
    # A*x + B*y + C = 0
    # X = points_full[:, 0]
    # Y = points_full[:, 1]
    # Incidence matrix: (56, 27)
    # val = A*X + B*Y + C
    
    X = points_full[:, 0].reshape(-1, 1)
    Y = points_full[:, 1].reshape(-1, 1)
    
    A = lines[:, 0].reshape(1, -1)
    B = lines[:, 1].reshape(1, -1)
    C = lines[:, 2].reshape(1, -1)
    
    vals = A * X + B * Y + C
    inc_matrix = (vals == 0).astype(np.int64)
    
    # Count incidences per point
    point_incs = inc_matrix.sum(axis=1)
    
    # 4. Remove 2 points with lowest incidence
    # If ties, remove any.
    # We want to keep 54 points.
    indices_to_remove = np.argsort(point_incs)[:2]
    
    # Create mask
    mask = np.ones(len(points_full), dtype=bool)
    mask[indices_to_remove] = False
    
    points = points_full[mask]
    
    # Ensure shape is (54, 2)
    if points.shape[0] != 54:
        # Fallback to simple grid if something went wrong
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)

    return points, lines

# EVOLVE-BLOCK-END
