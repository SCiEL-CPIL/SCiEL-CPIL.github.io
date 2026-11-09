import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Generate all weight-4 vectors
    # Indices of non-zero entries
    vecs = []
    for indices in itertools.combinations(range(D), 4):
        # 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(indices, signs):
                v[idx] = val
            vecs.append(v)
            
    vecs = np.array(vecs, dtype=np.int64)
    n = len(vecs)
    
    # Calculate dot products
    # dot[i, j] = vecs[i] . vecs[j]
    # We need to find a subset where dot[i, j] <= 2 for all i != j
    
    # Compute full dot product matrix
    # n is 5280. 5280^2 is ~27 million. 
    # This might take a few seconds in Python if not vectorized.
    # np.dot is vectorized.
    
    dot = vecs @ vecs.T
    
    # Conflict if dot > 2
    # Diagonal is 4. We ignore diagonal.
    conflict = (dot > 2)
    np.fill_diagonal(conflict, False)
    
    # Greedy Maximum Independent Set
    # Heuristic: Sort by degree (number of conflicts)
    degrees = conflict.sum(axis=1)
    
    # Try a few different orderings to see if we get a better set
    best_set = []
    best_size = 0
    
    # Ordering 1: Ascending degree
    order = np.argsort(degrees)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check if idx conflicts with any selected
        # selected_mask has True for selected indices
        # conflict[idx, selected_mask] should be all False
        if not np.any(conflict[idx, selected_mask]):
            selected.append(idx)
            selected_mask[idx] = True
            
    best_size = len(selected)
    best_set = selected
    
    # Ordering 2: Random shuffles (if time permits)
    # Given n=5280, the greedy is fast.
    # We can try a few random permutations.
    
    rng = np.random.default_rng(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        order = rng.permutation(n)
        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        
        for idx in order:
            if not np.any(conflict[idx, selected_mask]):
                selected.append(idx)
                selected_mask[idx] = True
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    # Convert back to vectors
    result = vecs[best_set]
    
    return result
