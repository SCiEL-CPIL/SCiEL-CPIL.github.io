"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Generate all weight 3 vectors
    # Support: 3 indices
    # Signs: 8 combinations
    w3_vectors = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(support, signs):
                v[idx] = val
            w3_vectors.append(v)
    
    # Generate all weight 2 vectors
    w2_vectors = []
    for support in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(support, signs):
                v[idx] = val
            w2_vectors.append(v)
            
    # Shuffle for better greedy performance
    np.random.shuffle(w3_vectors)
    np.random.shuffle(w2_vectors)
    
    selected = []
    
    def is_valid(v, current_list):
        # Check against all in current_list
        # v is int64 array
        # Norm squared of v is 3 or 2
        # R^2 is max norm squared.
        # If we only use w3 and w2, max norm squared is 3.
        # So we need dist^2 >= 3.
        for u in current_list:
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < 3:
                return False
        return True

    # Greedy add weight 3
    for v in w3_vectors:
        if time.time() > deadline:
            break
        if is_valid(v, selected):
            selected.append(v)
            
    # Greedy add weight 2
    for v in w2_vectors:
        if time.time() > deadline:
            break
        if is_valid(v, selected):
            selected.append(v)
            
    if not selected:
        # Fallback to seed if something went wrong or no time
        return np.array([[1,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
