import numpy as np
import itertools

D = 11

def construct():
    # Generate all vectors with exactly 4 non-zero entries (1 or -1)
    vectors = []
    indices = range(D)
    
    # Precompute supports and their vectors
    # To speed up conflict checking, we organize by support
    # support is a tuple of 4 indices
    # For each support, we have 16 vectors
    # We will store vectors as tuples for hashing? Or just indices and signs.
    
    all_vectors = []
    support_map = {} # support_tuple -> list of vector_indices in all_vectors
    
    idx = 0
    for comb in itertools.combinations(indices, 4):
        support = comb
        support_map[support] = []
        # Generate 16 sign patterns
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    # To check conflict between v1 and v2:
    # 1. If same support: conflict only if identical (same signs).
    # 2. If different support:
    #    inter = set(s1) & set(s2)
    #    if len(inter) < 3: no conflict.
    #    if len(inter) == 3:
    #       Check if signs on inter are identical.
    #    if len(inter) == 4: same support case.
    
    # Greedy selection
    selected_indices = []
    selected_vectors = [] # list of (support, signs)
    
    # We need fast lookup for neighbors.
    # For a given support S, neighbors are supports T with |S & T| = 3.
    # Precompute neighbors for each support?
    # 330 supports. 330 * 330 checks is small.
    neighbors = {s: [] for s in support_map.keys()}
    for s1 in support_map.keys():
        for s2 in support_map.keys():
            if s1 == s2: continue
            inter = len(set(s1) & set(s2))
            if inter == 3:
                neighbors[s1].append(s2)
                
    # Shuffle all vectors to break symmetry
    import random
    random.seed(42)
    shuffled_indices = list(range(len(all_vectors)))
    random.shuffle(shuffled_indices)
    
    # Data structure for fast conflict check
    # For each support, we maintain a set of selected sign patterns.
    # selected_signs[support] = set of signs tuples
    
    selected_signs = {s: set() for s in support_map.keys()}
    
    count = 0
    for idx in shuffled_indices:
        support, signs = all_vectors[idx][1], all_vectors[idx][2]
        
        # Check against same support
        if signs in selected_signs[support]:
            continue
            
        # Check against neighbors
        conflict = False
        for ns in neighbors[support]:
            # We need to check if any selected pattern in ns has the same signs on the intersection
            # Intersection I = set(support) & set(ns)
            # Let's precompute intersection sets? Or just compute.
            # Computing intersection of two 4-tuples is fast.
            inter = set(support) & set(ns)
            # Get signs of current vector on inter
            # signs is a tuple aligned with support.
            # We need to map support indices to signs.
            # This is a bit messy with tuples.
            
            # Let's change representation.
            # Vector is a dict {index: sign} or just the tuple aligned with support.
            # To compare on intersection, we need to know which positions in 'signs' correspond to 'inter'.
            
            # Optimization:
            # For each pair (support, neighbor), the intersection is fixed.
            # We can precompute for each neighbor pair (S, T) the mapping of indices.
            
            # Let's do a simpler check inside the loop, but optimized.
            # Since D=11, we can just reconstruct the values on the intersection.
            
            # Get values on intersection for current vector
            # Current vector values: {support[i]: signs[i]}
            # We need values on 'inter'.
            curr_vals = {}
            for i, s_val in zip(support, signs):
                if i in inter:
                    curr_vals[i] = s_val
            
            # Check against all selected patterns in ns
            # selected_signs[ns] contains tuples of signs aligned with ns.
            # We need to check if any of them matches curr_vals on inter.
            
            # To make this fast, we can precompute for each neighbor ns and intersection inter,
            # a mapping from "signature on inter" to count of selected vectors?
            # Or just iterate. Max 16 vectors in ns.
            
            for sel_signs in selected_signs[ns]:
                # Check if sel_signs matches curr_vals on inter
                match = True
                for i, s_val in zip(ns, sel_signs):
                    if i in inter:
                        if curr_vals[i] != s_val:
                            match = False
                            break
                if match:
                    conflict = True
                    break
            if conflict:
                break
                
        if not conflict:
            selected_signs[support].add(signs)
            selected_vectors.append(all_vectors[idx][3])
            count += 1
            
    if not selected_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(selected_vectors, dtype=np.int64)
