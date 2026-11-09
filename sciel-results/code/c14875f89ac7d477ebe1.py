# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    Uses a 6x9 grid and vectorized incidence counting to select the top 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    # Build the 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate candidate lines
    # We will test lines Ax + By + C = 0
    # For a grid, the best lines are those aligned with the grid or diagonals.
    # We generate all lines with small A, B and check C values that pass through at least one point.
    
    candidate_lines = []
    seen = set()
    
    # Helper to add a line if valid and unique
    def add_line(A, B, C):
        # Normalize
        if A == 0 and B == 0:
            return
        # Make first non-zero positive
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        # Reduce by GCD
        g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        key = (A, B, C)
        if key not in seen:
            seen.add(key)
            candidate_lines.append((A, B, C))

    # Vertical lines: x = c
    for c in range(1, 7):
        add_line(1, 0, -c)
        
    # Horizontal lines: y = c
    for c in range(1, 10):
        add_line(0, 1, -c)
        
    # Diagonal and other slopes
    # A, B small integers
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            if A == 0 and B == 1: # Already added horizontals
                continue
            if A == 1 and B == 0: # Already added verticals
                continue
                
            # Find C values such that the line passes through at least one grid point
            # A*x + B*y + C = 0 => C = -(A*x + B*y)
            # We only need to check C values that result in at least 1 point.
            # Actually, we can just iterate through all points and compute C, then unique them.
            Cs = np.unique(-(A * points[:, 0] + B * points[:, 1]))
            for C in Cs:
                add_line(A, B, int(C))
                
    # Count incidences for all candidate lines
    # points shape (54, 2)
    # candidate_lines is list of (A, B, C)
    
    incidences = []
    for A, B, C in candidate_lines:
        # Vectorized check
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        if count > 0:
            incidences.append((count, A, B, C))
            
    # Sort by incidence count descending
    incidences.sort(key=lambda x: -x[0])
    
    # Select top 27
    selected = incidences[:27]
    
    lines = np.array([(A, B, C) for _, A, B, C in selected], dtype=np.int64)
    
    # Verify
    total = 0
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
        
    return points, lines

# EVOLVE-BLOCK-END
