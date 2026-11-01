import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all weight 4 vectors
    # Entries are 1 or -1 at 4 positions, 0 elsewhere
    vectors = []
    indices = range(D)
    
    # Pre-compute all combinations of 4 indices
    supports = list(itertools.combinations(indices, 4))
    
    # For each support, generate all 2^4 = 16 sign variations
    # We can generate them on the fly or pre-store
    # To save memory, we process one support at a time in the greedy loop?
    # No, we need to check against ALL existing vectors.
    # Let's generate all 5280 vectors first.
    all_vecs = []
    for supp in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(supp, signs):
                v[idx] = val
            all_vecs.append(v)
            
    all_vecs = np.array(all_vecs)
    np.random.shuffle(all_vecs)
    
    selected = []
    
    # Greedy selection
    for v in all_vecs:
        if time.time() > deadline:
            break
            
        # Check if v is valid against all selected
        # Dist sq >= 4
        if not selected:
            selected.append(v)
            continue
            
        # Vectorized check
        # selected is list of arrays
        # Convert to array for speed?
        # For small selected, loop is fine. For large, array is better.
        
        # Let's use a loop for simplicity and safety, breaking early
        valid = True
        for u in selected:
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < 4:
                valid = False
                break
                
        if valid:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
