"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    start = time.time()
    deadline = start + 130

    # Generate all integer vectors of norm^2 = 4 with at most 4 nonzero entries
    candidates = []
    
    # Type A: one ±2 (22 vectors)
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s*2
            candidates.append(v)
    
    # Type B: two ±1's (220 vectors)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type C: four ±1's (5280 vectors)
    for combo in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = [0]*D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    # Greedy independent set with dot <= 2 constraint (norm^2=4 vectors)
    selected = []
    n = len(candidates)
    
    # Process in order; for speed, use numpy for batch checks when selected is large
    for v in candidates:
        if time.time() > deadline - 5:
            break
        if not selected:
            selected.append(v)
            continue
        # Check compatibility
        # Convert to array for vectorized dot
        v_arr = np.array(v)
        sel_arr = np.array(selected)
        dots = sel_arr @ v_arr
        if np.all(dots <= 2):
            selected.append(v)
    
    result = np.array(selected, dtype=np.int64)
    if len(result) == 0:
        result = np.array([[1] + [0]*(D-1)], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
