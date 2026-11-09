# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import Counter

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 630.0
    
    # The optimal grid for 54 points to maximize incidences with 27 lines
    # is a 6x9 grid (or 9x6), as it maximizes the minimum dimension (6),
    # allowing for diagonals of length 6. Other factorizations like 3x18
    # allow max diagonal length 3, which is suboptimal.
    # We use 6x9.
    nx, ny = 6, 9
    
    xs = np.arange(nx)
    ys = np.arange(ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    candidate_lines = []
    
    # 1. Vertical lines: x = c -> (1, 0, -c)
    for c in range(nx):
        candidate_lines.append((1, 0, -c))
        
    # 2. Horizontal lines: y = c -> (0, 1, -c)
    for c in range(ny):
        candidate_lines.append((0, 1, -c))
        
    # 3. Diagonal lines with primitive slopes
    # Line: dy*x - dx*y + C = 0
    # We iterate over small integer dx, dy with gcd(dx, dy) = 1
    # To cover all relevant slopes, we limit dx, dy to the grid dimensions.
    # dx, dy can be negative.
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    # Generate unique primitive directions
    # We can restrict dx > 0 to avoid duplicates (A,B,C) ~ (-A,-B,-C)
    # But we need to handle vertical/horizontal separately or include them.
    # Vertical: dx=0, dy=1 (handled above)
    # Horizontal: dx=1, dy=0 (handled above)
    
    # For diagonals, dx, dy != 0
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, dy) != 1:
                continue
            # Normalize direction: dx > 0
            # If dx < 0, multiply by -1. We only iterate dx > 0.
            # So we cover all slopes with positive dx.
            
            # Collect C values
            c_counts = Counter()
            # Vectorized calculation of C = dx*y - dy*x for all points
            # C = dx * points[:, 1] - dy * points[:, 0]
            C_vals = dx * points[:, 1] - dy * points[:, 0]
            # Count occurrences
            # Use np.unique for speed
            vals, counts = np.unique(C_vals, return_counts=True)
            for C, count in zip(vals, counts):
                if count >= 2:
                    # Line: dy*x - dx*y + C = 0
                    # A=dy, B=-dx, C=C
                    candidate_lines.append((dy, -dx, C))
                    
    # Normalize and deduplicate lines
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        # Compute GCD of absolute values
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        # Make first non-zero positive
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
        
    seen = set()
    unique_candidates = []
    for line in candidate_lines:
        norm = normalize_line(*line)
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(line)
            
    # Calculate incidences for each unique candidate
    # points: (54, 2)
    # lines: (M, 3)
    # Incidence: A*x + B*y + C == 0
    
    incidences = []
    for A, B, C in unique_candidates:
        # Vectorized check
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        incidences.append(count)
        
    incidences = np.array(incidences)
    
    # Select top 27 lines
    if len(unique_candidates) >= 27:
        # Sort by incidence descending
        top_indices = np.argsort(-incidences)[:27]
    else:
        top_indices = np.arange(len(unique_candidates))
        # Pad if necessary (should not happen with 6x9 grid)
        while len(top_indices) < 27:
            top_indices = np.append(top_indices, 0)
            
    selected_lines = np.array([unique_candidates[i] for i in top_indices], dtype=np.int64)
    
    # Final verification and cleanup
    # Ensure exactly 27 unique lines
    final_lines = []
    seen_final = set()
    for line in selected_lines:
        norm = normalize_line(*line)
        if norm not in seen_final:
            seen_final.add(norm)
            final_lines.append(line)
            
    # If we have fewer than 27 (due to duplicates in top 27?), add more
    if len(final_lines) < 27:
        for idx in range(len(unique_candidates)):
            if len(final_lines) >= 27:
                break
            norm = normalize_line(*unique_candidates[idx])
            if norm not in seen_final:
                seen_final.add(norm)
                final_lines.append(unique_candidates[idx])
                
    lines = np.array(final_lines[:27], dtype=np.int64)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        pass # Return current best
        
    return points, lines

# EVOLVE-BLOCK-END
