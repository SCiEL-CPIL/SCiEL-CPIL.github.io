import numpy as np
import time

def construct():
    D = 11
    deadline = time.time() + 110  # Safety margin
    
    # Generate all integer vectors with norm squared 4
    # Type 1: 2e_i (22 vectors)
    # Type 2: 4 entries of +/-1 (5280 vectors)
    
    vectors = []
    
    # Type 1
    for i in range(D):
        v = [0]*D
        v[i] = 2
        vectors.append(v)
        v[i] = -2
        vectors.append(v)
        
    # Type 2
    import itertools
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    
    # Greedy selection
    selected = []
    n = len(vectors)
    # Sort by norm? All norm 4.
    # Maybe randomize order to break symmetry?
    # Or just pick in order.
    
    # To speed up, we can check distance to selected vectors.
    # If selected grows large, checking all pairs is O(k^2 * D).
    # k ~ 5000, D=11. 25e6 * 11 ~ 2.75e8 ops. Might be slow in Python.
    # Use numpy for distance calculation?
    # Batch processing:
    
    # Let's try a simpler greedy:
    # Pick first vector.
    # For next vector, check dist to all selected.
    # If min dist >= 2, add.
    
    # Optimization:
    # We can precompute dot products?
    
    # Given the risk of timeout, and 220 is a solid baseline,
    # I will try to run this greedy for a short time.
    # If it exceeds 220, great. If not, return 220 roots.
    
    # Let's fallback to D11 roots if greedy is too slow or fails.
    
    # Generate D11 roots
    d11_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                d11_vectors.append(v)
    d11_array = np.array(d11_vectors, dtype=np.int64)
    
    # Try greedy on norm 4 vectors
    # To make it faster, we can just check against a subset or use a better algo.
    # But for now, let's just return D11 roots if we don't have a better one quickly.
    
    # Actually, let's just return D11 roots to be safe?
    # No, the prompt asks for improvement.
    # I will implement a quick greedy check.
    
    selected_idx = [0]
    current_v = vectors[0]
    
    for i in range(1, n):
        if time.time() > deadline - 10:
            break
        v = vectors[i]
        # Check dist to last few selected? No, must check all.
        # This is O(n*k).
        # Let's use a vectorized check against the selected array.
        if len(selected_idx) > 0:
            sel = vectors[selected_idx]
            diffs = v - sel
            dists = np.sum(diffs**2, axis=1)
            if np.min(dists) >= 4:
                selected_idx.append(i)
                
    if len(selected_idx) > len(d11_array):
        return vectors[selected_idx]
    else:
        return d11_array
