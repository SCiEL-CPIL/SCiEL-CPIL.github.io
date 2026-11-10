import numpy as np
import time
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 625.0  # safety margin
    
    # Try 7x8 grid (56 points), remove 2
    nx, ny = 7, 8
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points)
    
    # Generate candidate lines
    candidates = set()
    
    # Vertical
    for c in range(nx):
        candidates.add((1, 0, -c))
        
    # Horizontal
    for c in range(ny):
        candidates.add((0, 1, -c))
        
    # Diagonals
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            
            # A=dy, B=-dx
            # C = dx*y - dy*x
            C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
            vals = np.unique(C_vals)
            for C in vals:
                candidates.add((dy, -dx, C))
                
    # Normalize lines
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
        
    unique_lines = []
    for line in candidates:
        norm = normalize_line(*line)
        if norm:
            unique_lines.append(norm)
            
    unique_lines = np.array(unique_lines, dtype=np.int64)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Iterate over pairs of points to remove
    # There are 56 points, so 56*55/2 = 1540 pairs
    for i, j in combinations(range(n_pts), 2):
        if time.time() > deadline:
            break
            
        # Create mask
        mask = np.ones(n_pts, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        active_points = base_points[mask]
        
        # Calculate incidences
        # A*x + B*y + C == 0
        # unique_lines: (M, 3)
        # active_points: (54, 2)
        
        # Vectorized calculation
        # For each line, sum( A*px + B*py + C == 0 )
        # This is (M, 54) boolean array.
        
        # To save memory/time, we can do it in chunks or just use einsum/broadcast
        # M is ~200, 54 is small.
        
        A = unique_lines[:, 0]
        B = unique_lines[:, 1]
        C = unique_lines[:, 2]
        
        x = active_points[:, 0]
        y = active_points[:, 1]
        
        # vals[m, k] = A[m]*x[k] + B[m]*y[k] + C[m]
        vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
        incidences = np.sum(vals == 0, axis=1)
        
        # Select top 27
        if len(incidences) >= 27:
            top_indices = np.argsort(-incidences)[:27]
            score = np.sum(incidences[top_indices])
        else:
            score = np.sum(incidences)
            top_indices = np.arange(len(incidences))
            
        if score > best_score:
            best_score = score
            best_points = active_points.copy()
            best_lines = unique_lines[top_indices].copy()
            
    # Fallback if no pair found or time out
    if best_points is None:
        best_points = base_points[:54]
        # Calculate lines for this set
        A = unique_lines[:, 0]
        B = unique_lines[:, 1]
        C = unique_lines[:, 2]
        x = best_points[:, 0]
        y = best_points[:, 1]
        vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
        incidences = np.sum(vals == 0, axis=1)
        top_indices = np.argsort(-incidences)[:27]
        best_lines = unique_lines[top_indices]
        
    return best_points, best_lines
