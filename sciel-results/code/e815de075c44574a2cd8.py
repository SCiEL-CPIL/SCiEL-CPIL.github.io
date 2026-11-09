# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a 6x9 rectangular grid and selects the 27 lines with the highest incidence counts.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, use 630s wall clock but keep margin
    
    # 1. Define the 6x9 grid points
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Find all unique lines passing through at least 2 points
    # Line representation: (A, B, C) normalized such that gcd(|A|,|B|,|C|)=1
    # and first non-zero of A,B is positive.
    
    lines_dict = {}
    
    n_points = len(points)
    # Iterate over pairs of points to find lines
    # To avoid O(N^3) counting, we can first find candidate lines, then count.
    # With 54 points, N^2 = 2916 pairs. This is small.
    
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for A*x + B*y + C = 0
            # (y2 - y1) * x - (x2 - x1) * y + (x2*y1 - x1*y2) = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = x1 * y2 - x2 * y1
            
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            A //= g
            B //= g
            C //= g
            
            # Make first non-zero positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key not in lines_dict:
                lines_dict[key] = 0
                
    # 3. Count incidences for each unique line
    # To speed this up, we can just iterate over points for each line.
    # Number of lines is roughly N^2 = 3000.
    # 3000 * 54 = 162,000 operations. Very fast.
    
    line_counts = []
    for (A, B, C) in lines_dict.keys():
        count = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                count += 1
        line_counts.append((count, (A, B, C)))
        
    # 4. Sort lines by count descending
    line_counts.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Pick top 27 lines
    selected_lines = []
    for count, line in line_counts:
        if len(selected_lines) == 27:
            break
        selected_lines.append(line)
        
    lines = np.array(selected_lines, dtype=np.int64)
    
    # Check time
    if time.time() > deadline:
         # Fallback to simple grid if somehow slow (unlikely)
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
