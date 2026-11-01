"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate candidate vectors with norm squared 4 (4 non-zero entries of +/- 1)
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    selected = []
    selected_arr = [] # List of arrays for fast append, then stack?
    # Better: maintain a list of vectors, check against all previous.
    # To speed up, we can check in batches or just use a simple loop with numpy dot.
    
    # Optimization: Pre-compute dot products is too expensive for all pairs.
    # Greedy:
    # For each candidate, compute dot with all selected.
    # If max dot <= 2, add it.
    
    # We expect to select ~1000-2000 vectors.
    # 5280 * 2000 * 11 operations ~ 116M. Might be slow in Python loop.
    # Use NumPy vectorization for the check.
    
    selected_list = []
    
    for v in candidates:
        if time.time() > deadline:
            break
        
        if not selected_list:
            selected_list.append(v)
            continue
            
        # Convert selected_list to array for dot product
        # This is inefficient if done every time.
        # Better: maintain a numpy array `selected_arr` and a counter.
        
    # Let's restructure for speed.
    selected_arr = np.empty((0, D), dtype=np.int8)
    
    for v in candidates:
        if time.time() > deadline:
            break
            
        if selected_arr.shape[0] == 0:
            selected_arr = np.vstack([v])
            continue
            
        # Compute dot products of v with all selected
        dots = selected_arr.dot(v)
        if np.max(dots) <= 2:
            selected_arr = np.vstack([selected_arr, v])
            
    if selected_arr.shape[0] == 0:
        # Fallback to D11 vectors if something failed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    vec = [0] * D
                    vec[i] = si
                    vec[j] = sj
                    vectors.append(vec)
        return np.array(vectors, dtype=np.int64)
        
    return selected_arr.astype(np.int64)
# EVOLVE-BLOCK-END
