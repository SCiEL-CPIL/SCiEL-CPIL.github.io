"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 120s

    # Generate weight-2 vectors: C(11,2)*4 = 220 vectors, norm = 2
    # Generate weight-3 vectors: C(11,3)*8 = 1320 vectors, norm = sqrt(3)
    # To combine, we need same norm. Let's scale:
    # Weight-2: norm 2, need dot <= 1
    # Weight-3: norm sqrt(3), need dot <= 1.5 -> dot <= 1 (integer)
    # But different norms can't be directly combined under the same constraint.
    # 
    # Alternative: use only one weight, or find a common scaling.
    # Let's try weight-3 vectors which give more candidates.
    
    # Strategy: Generate weight-3 vectors, find large independent set
    # Weight-3: norm = sqrt(3), conflict if dot > 1 (i.e., dot >= 2)
    
    vectors_list = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors_list.append(v)
    
    vectors = np.array(vectors_list, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute conflict matrix: conflict if dot > 1 (i.e., dot >= 2)
    G = vectors @ vectors.T
    conflict = G >= 2  # dot >= 2 means conflict for weight-3 (need dot <= 1)
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set with min-degree selection
    in_set = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    
    while remaining.any():
        if time.time() > deadline:
            break
        
        # Find vertex with minimum degree among remaining
        masked_degrees = np.where(remaining, conflict.sum(axis=1), np.inf)
        idx = np.argmin(masked_degrees)
        
        if not remaining[idx]:
            break
        
        in_set[idx] = True
        remaining[idx] = False
        
        # Remove neighbors
        neighbors = conflict[idx] & remaining
        remaining[neighbors] = False
    
    result = vectors[in_set]
    
    # Try to improve with local search: try adding removed vertices
    # This is a simple improvement step
    if time.time() < deadline - 10:
        # Try to add back some vertices that were removed
        added = in_set.copy()
        for i in range(n):
            if time.time() > deadline - 5:
                break
            if added[i]:
                continue
            # Check if i can be added: no conflict with any in added set
            if not (conflict[i] & added).any():
                added[i] = True
        
        if added.sum() > in_set.sum():
            in_set = added
            result = vectors[in_set]
    
    return result
# EVOLVE-BLOCK-END
