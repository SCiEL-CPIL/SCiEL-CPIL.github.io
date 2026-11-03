import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all 4-subsets of {0, ..., 10}
    subsets = []
    for comb in itertools.combinations(range(D), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        subsets.append(mask)
    
    n_subsets = len(subsets)
    
    # Precompute compatibility? 
    # 330x330 matrix is small.
    # compat[i][j] = 1 if popcount(subsets[i] & subsets[j]) <= 2
    
    # We want Max Independent Set in incompatibility graph.
    # Or Max Clique in compatibility graph.
    
    best_k = 0
    best_vectors = []
    
    # Greedy with random restarts
    import random
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        random.shuffle(subsets)
        selected = []
        
        for mask in subsets:
            # Check compatibility with all in selected
            compatible = True
            for sel_mask in selected:
                # Intersection size
                inter = mask & sel_mask
                # Count bits
                # Python 3.10+ has int.bit_count()
                if inter.bit_count() > 2:
                    compatible = False
                    break
            
            if compatible:
                selected.append(mask)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_vectors = selected
            
        # Optional: Local search to improve?
        # Given time, greedy might be enough.
        
    # Construct vectors
    vectors = []
    for mask in best_vectors:
        # Find indices
        indices = []
        for i in range(D):
            if mask & (1 << i):
                indices.append(i)
        
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        # Fallback to seed
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
