import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate all vectors of weight 4 (4 non-zero entries, all +/- 1)
    vectors = []
    supports = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
            supports.append(support)
            
    n_vecs = len(vectors)
    
    # Pre-compute conflict index
    # Key: (tuple(triple), tuple(signs_on_triple))
    # Value: list of indices of vectors having this triple and these signs
    conflict_index = {}
    
    # Map support to a frozenset or tuple for quick comparison
    # Actually, we just need to check if support is same.
    
    for i, v in enumerate(vectors):
        sup = supports[i]
        # Generate all 4 triples from this support
        for triple in itertools.combinations(sup, 3):
            # Get signs on these triple indices
            # sup is a tuple of 4 indices. triple is a tuple of 3.
            # We need to map the signs correctly.
            # v is a list of length D.
            signs_on_triple = tuple(v[idx] for idx in triple)
            key = (triple, signs_on_triple)
            if key not in conflict_index:
                conflict_index[key] = []
            conflict_index[key].append(i)
            
    # Greedy selection
    best_count = 0
    best_vectors = []
    
    # Run multiple times with different shuffles
    rng = np.random.default_rng(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        order = np.arange(n_vecs)
        rng.shuffle(order)
        
        selected_indices = set()
        current_count = 0
        
        for idx in order:
            v = vectors[idx]
            sup = supports[idx]
            
            is_conflict = False
            
            # Check 4 triples of the current vector's support
            for triple in itertools.combinations(sup, 3):
                signs_on_triple = tuple(v[idx_val] for idx_val in triple)
                key = (triple, signs_on_triple)
                
                if key in conflict_index:
                    candidates = conflict_index[key]
                    for cand_idx in candidates:
                        if cand_idx in selected_indices:
                            # If it's the same vector, it's not a conflict (but we haven't added it yet, so it won't be in set)
                            # If it's a different vector with same support, it's NOT a conflict.
                            # If it's a different vector with different support, it IS a conflict.
                            if cand_idx == idx:
                                continue
                            if supports[cand_idx] == sup:
                                continue
                            is_conflict = True
                            break
                if is_conflict:
                    break
                    
            if not is_conflict:
                selected_indices.add(idx)
                current_count += 1
                
        if current_count > best_count:
            best_count = current_count
            best_vectors = [vectors[i] for i in selected_indices]
            
    if not best_vectors:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)
