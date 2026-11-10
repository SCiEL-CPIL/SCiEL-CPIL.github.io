# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from collections import Counter

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 625.0  # safety margin
    
    # Use a 6x9 grid to maximize incidences
    nx, ny = 6, 9
    
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate all candidate lines with >= 2 points
    candidates = []
    
    # Vertical lines: x = c -> (1, 0, -c)
    for c in range(nx):
        candidates.append((1, 0, -c))
        
    # Horizontal lines: y = c -> (0, 1, -c)
    for c in range(ny):
        candidates.append((0, 1, -c))
        
    # Diagonal lines: for all primitive (dx, dy) with dx > 0, dy != 0
    # Line equation: dy*x - dx*y + C = 0, i.e., A=dy, B=-dx, C=C
    # For each such direction, find all C values that give >= 2 incidences
    
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            
            # C = dx*y - dy*x for each point
            C_vals = dx * points[:, 1] - dy * points[:, 0]
            vals, counts = np.unique(C_vals, return_counts=True)
            for C, count in zip(vals, counts):
                if count >= 2:
                    candidates.append((dy, -dx, C))
    
    # Normalize and deduplicate
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    seen = set()
    unique_candidates = []
    for line in candidates:
        norm = normalize_line(*line)
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(line)
    
    # Compute incidences for each unique candidate
    incidences = np.zeros(len(unique_candidates), dtype=np.int64)
    for i, (A, B, C) in enumerate(unique_candidates):
        vals = A * points[:, 0] + B * points[:, 1] + C
        incidences[i] = np.sum(vals == 0)
    
    # Select top 27 lines
    if len(unique_candidates) >= 27:
        top_indices = np.argsort(-incidences)[:27]
    else:
        top_indices = np.arange(len(unique_candidates))
    
    selected_lines = [unique_candidates[i] for i in top_indices]
    
    # Final dedup check
    final_lines = []
    seen_final = set()
    for line in selected_lines:
        norm = normalize_line(*line)
        if norm not in seen_final:
            seen_final.add(norm)
            final_lines.append(line)
    
    # If we have fewer than 27, add more
    if len(final_lines) < 27:
        for i in range(len(unique_candidates)):
            if len(final_lines) >= 27:
                break
            norm = normalize_line(*unique_candidates[i])
            if norm not in seen_final:
                seen_final.add(norm)
                final_lines.append(unique_candidates[i])
    
    lines = np.array(final_lines[:27], dtype=np.int64)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # Return current best
    
    return points, lines

# EVOLVE-BLOCK-END
