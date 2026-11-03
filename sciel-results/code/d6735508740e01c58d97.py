import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Hex grid basis for flat-top unit hexagons
    # Adjacent centers: distance √3 apart
    # Basis vectors: (√3, 0) and (√3/2, 3/2)
    sqrt3 = np.sqrt(3)
    ax = np.array([sqrt3, 0.0])
    ay = np.array([sqrt3/2, 1.5])
    
    # Generate candidate packings as sets of (col, row) in hex grid coords
    # Convert to cartesian: pos = col * ax + row * ay
    
    def hex_to_cart(col, row):
        return col * ax + row * ay
    
    # Try various configurations of 11 cells
    # Use axial coordinates (q, r) where cart = q*ax + r*ay
    # Neighbors of (q,r): (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    
    def neighbors(q, r):
        return [(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)]
    
    # Generate many candidate sets of 11 cells using BFS/greedy from a seed
    candidates = []
    
    # Predefined good patterns
    # Pattern 1: 3-4-4 (rows of 3,4,4)
    # In axial coords, let's build row by row
    def build_pattern(rows_counts, offset_pattern=None):
        """Build a hex grid pattern with given row counts."""
        cells = []
        r = 0
        for i, count in enumerate(rows_counts):
            # Center this row
            start_q = -(count - 1) // 2
            for j in range(count):
                q = start_q + j
                cells.append((q, r))
            r += 1
        return cells
    
    # Various row patterns that sum to 11
    row_patterns = [
        [3, 4, 4],
        [4, 4, 3],
        [2, 4, 5],
        [5, 4, 2],
        [3, 3, 5],
        [5, 3, 3],
        [4, 3, 4],
        [3, 5, 3],
        [1, 5, 5],
        [5, 5, 1],
        [2, 3, 6],
        [6, 3, 2],
        [1, 4, 6],
        [6, 4, 1],
        [2, 5, 4],
        [4, 5, 2],
        [1, 3, 7],
        [7, 3, 1],
        [3, 2, 6],
        [6, 2, 3],
        [2, 2, 7],
        [7, 2, 2],
        [1, 2, 8],
        [8, 2, 1],
        [4, 2, 5],
        [5, 2, 4],
        [3, 1, 7],
        [7, 1, 3],
        [2, 1, 8],
        [8, 1, 2],
        [1, 1, 9],
        [9, 1, 1],
        [0, 5, 6],
        [6, 5, 0],
        [1, 6, 4],
        [4, 6, 1],
        [2, 6, 3],
        [3, 6, 2],
        [1, 7, 3],
        [3, 7, 1],
        [2, 7, 2],
        [0, 6, 5],
        [5, 6, 0],
        [1, 5, 5],
        [5, 5, 1],
        [2, 4, 5],
        [5, 4, 2],
        [3, 4, 4],
        [4, 4, 3],
        [4, 3, 4],
        [3, 5, 3],
        [2, 3, 6],
        [6, 3, 2],
        [1, 4, 6],
        [6, 4, 1],
    ]
    
    for pattern in row_patterns:
        cells = build_pattern(pattern)
        if len(cells) == N:
            candidates.append(cells)
    
    # Also try some manually designed compact patterns
    # Compact 11-hexagon: center + ring of 6 + 4 more
    # Axial: (0,0), (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1), then add (2,-1),(2,-2),(1,-2),(0,-2)
    manual_patterns = [
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(2,-2),(1,-2),(0,-2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(1,-2),(0,-2),(-1,-2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(-2,1),(-1,2),(0,2),(1,1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(3,-1),(3,-2),(2,-2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(-2,0),(-3,1),(-3,2),(-2,2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(3,-2),(2,-2),(1,-2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(1,-2),(0,-2),(-1,-2)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(-2,1),(-3,2),(-2,2),(-1,2)],
    ]
    for mp in manual_patterns:
        if len(mp) == N:
            candidates.append(mp)
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    candidates = unique_candidates
    
    best_S = float('inf')
    best_result = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Convert to cartesian
        centers = np.array([hex_to_cart(q, r) for q, r in cand])
        
        # Center the configuration
        centers = centers - centers.mean(axis=0)
        
        # For each rotation angle of the outer hexagon, find minimal S
        # The outer hexagon can rotate; we try angles 0 to 30 degrees (symmetry)
        # For a given outer rotation θ, the constraint for center (x,y) is:
        # Rotate point by -θ, then check max(|x'|, |√3*x'+y'|/√3, |√3*x'-y'|/√3) ≤ S
        
        def compute_S(centers, theta):
            """Compute minimal S for enclosing hexagon rotated by theta."""
            c = np.cos(theta)
            s = np.sin(theta)
            # Rotate centers by -theta
            rot = np.column_stack([
                c * centers[:,0] + s * centers[:,1],
                -s * centers[:,0] + c * centers[:,1]
            ])
            x = rot[:,0]
            y = rot[:,1]
            # Constraints: |x|, |√3*x+y|/√3, |√3*x-y|/√3
            val1 = np.abs(x)
            val2 = np.abs(sqrt3*x + y) / sqrt3
            val3 = np.abs(sqrt3*x - y) / sqrt3
            return np.maximum.reduce([val1, val2, val3]).max()
        
        # Try multiple rotation angles
        best_local_S = float('inf')
        best_local_theta = 0.0
        
        # Coarse search
        for theta_deg in range(0, 31, 1):
            theta = np.radians(theta_deg)
            S_val = compute_S(centers, theta)
            if S_val < best_local_S:
                best_local_S = S_val
                best_local_theta = theta
        
        # Fine search around best
        if best_local_S < best_S:
            for theta_deg in np.arange(best_local_theta - 0.5, best_local_theta + 0.5, 0.05):
                theta = np.radians(theta_deg)
                S_val = compute_S(centers, theta)
                if S_val < best_local_S:
                    best_local_S = S_val
                    best_local_theta = theta
        
        if best_local_S < best_S:
            best_S = best_local_S
            # Compute the actual inner positions with this rotation
            theta = best_local_theta
            c = np.cos(theta)
            s = np.sin(theta)
            # The inner hexagons are placed at 'centers' in the rotated frame
            # In the original frame, they are at 'centers' (since we only rotated the outer)
            # Actually, the inner hexagons have angle 0 in the local frame
            # We need to express everything in a consistent frame
            
            # Let's work in the frame where the outer hexagon is at angle 0
            # Inner centers are at 'centers' (already centered)
            # Inner hexagons have angle 0
            inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
            best_result = (inner, (0.0, 0.0), best_S, 0.0)
    
    # If no valid result found, use a fallback
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        # Compute S for this
        centers = np.array([[sqrt3*k, 0.0] for k in range(-(N//2), N - N//2)])
        centers = centers - centers.mean(axis=0)
        S_val = compute_S(centers, 0.0)
        best_result = (inner, (0.0, 0.0), S_val, 0.0)
    
    return best_result
