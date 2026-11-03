import numpy as np
import itertools
import random
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate all vectors in the m=4 shell
    # Supports are combinations of 4 indices from 11
    # Signs are 4 bits (16 patterns)
    vectors = []
    supports = []
    for comb in itertools.combinations(range(D), 4):
        supports.append(comb)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            vectors.append(v)
    
    n_vecs = len(vectors)
    
    # Pre-compute 3-subsets and signs for each vector to speed up conflict check
    # For each vector, store list of (key, support)
    # Key: (sorted_tuple_of_3_indices, tuple_of_signs_on_those_indices)
    vec_data = []
    for i, v in enumerate(vectors):
        sup = supports[i // 16] # This mapping is fragile if we change generation.
        # Better to store support with vector
        pass

    # Let's regenerate with support stored
    vec_list = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            vec_list.append((v, comb))
            
    n_vecs = len(vec_list)
    
    # Pre-compute keys for each vector
    # Key is (tuple(sorted(T)), tuple(signs))
    precomputed_keys = []
    for v, comb in vec_list:
        keys = []
        # 4 subsets of size 3
        for i in range(4):
            # Exclude index i from comb
            T = tuple(sorted([comb[j] for j in range(4) if j != i]))
            # Get signs for these indices
            # We need to map indices in T to signs
            # comb is (c0, c1, c2, c3)
            # signs is (s0, s1, s2, s3)
            # If we exclude i, we take signs at j != i
            t_signs = []
            for idx in T:
                # Find position of idx in comb
                pos = comb.index(idx)
                t_signs.append(vec_list[0][1][pos] if False else None) # Placeholder
            # Simpler: just iterate
            t_signs = []
            for idx in T:
                pos = comb.index(idx)
                # Get sign from the original signs tuple
                # We lost the signs tuple in vec_list. Let's keep it.
            pass
            
    # Let's redo vec_list to include signs tuple
    vec_list = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            vec_list.append((v, comb, signs))
            
    precomputed_keys = []
    for v, comb, signs in vec_list:
        keys = []
        for i in range(4):
            # Exclude i-th element of comb
            T_indices = [comb[j] for j in range(4) if j != i]
            T_signs = [signs[j] for j in range(4) if j != i]
            # Sort by index to make key canonical
            paired = sorted(zip(T_indices, T_signs))
            T_tuple = tuple(p[0] for p in paired)
            S_tuple = tuple(p[1] for p in paired)
            keys.append((T_tuple, S_tuple))
        precomputed_keys.append(keys)
        
    best_set = []
    best_size = 0
    
    # Time guard
    start_time = time.time()
    timeout = 110 # Safety margin
    
    for restart in range(10000):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(vec_list)
        
        used_keys = {} # Key -> Set of Supports
        current_set = []
        
        for v, comb, signs in vec_list:
            conflict = False
            for key in precomputed_keys[vec_list.index((v, comb, signs))]: # This index lookup is slow!
                # We need to iterate over the precomputed keys corresponding to the current item in the shuffled list.
                # Since we shuffled vec_list, we lost the link to precomputed_keys.
                # Let's shuffle indices instead.
                pass

    # Correct Approach:
    # 1. Generate vec_list and precomputed_keys.
    # 2. Create indices 0..N-1.
    # 3. Shuffle indices.
    # 4. Iterate indices.
    
    indices = list(range(n_vecs))
    
    for restart in range(5000):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(indices)
        
        used_keys = {}
        current_set = []
        
        for idx in indices:
            v, comb, signs = vec_list[idx]
            keys = precomputed_keys[idx]
            
            conflict = False
            for key in keys:
                if key in used_keys:
                    # Check if any support in used_keys[key] is different from comb
                    # If yes, conflict.
                    for sup in used_keys[key]:
                        if sup != comb:
                            conflict = True
                            break
                    if conflict:
                        break
            
            if not conflict:
                current_set.append(v)
                # Update used_keys
                for key in keys:
                    if key not in used_keys:
                        used_keys[key] = set()
                    used_keys[key].add(comb)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
# EVOLVE-BLOCK-END
