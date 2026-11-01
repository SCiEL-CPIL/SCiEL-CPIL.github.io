import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate all vectors of norm sqrt(3)
    # 3 indices with +/- 1
    candidates = []
    for indices in itertools.combinations(range(D), 3):
        i, j, k = indices
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    candidates = np.array(candidates)
    # Remove duplicates? No, combinations are unique.
    
    selected = []
    # Sort candidates to prioritize? 
    # Maybe just iterate.
    
    for v in candidates:
        if time.time() > deadline:
            break
            
        # Check compatibility with all selected vectors
        # Dot product must be <= 1
        # Since selected are also norm sqrt(3), 
        # dist^2 = 3+3 - 2 dot = 6 - 2 dot.
        # dist >= sqrt(3) => dist^2 >= 3 => 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.5 => dot <= 1.
        
        if not selected:
            selected.append(v)
            continue
            
        # Vectorized dot product
        # selected is list of arrays
        sel_arr = np.array(selected)
        dots = sel_arr @ v
        if np.all(dots <= 1):
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
