import numpy as np
import itertools
import time
from collections import defaultdict

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    # Type 1: One 2, rest 0: ±2e_i
    type1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1.append(v)
    
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    type2 = []
    type2_info = [] # (indices, signs)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            type2.append(v)
            type2_info.append((idxs, signs))
            
    type1 = np.array(type1, dtype=np.int64)
    type2 = np.array(type2, dtype=np.int64)
    N2 = len(type2)
    
    # Build conflict graph for type2
    # Conflict if dot product > 2
    # We know conflicts only if |support intersection| == 3 and signs match on intersection.
    # Let's build adjacency list.
    adj = defaultdict(set)
    
    # Map vector index to (indices, signs)
    # We already have type2_info.
    
    # To find conflicts efficiently, we can iterate through all pairs? No, N2=5280, pairs ~14M. Too slow in Python.
    # Use the structural insight.
    # For each vector u (index i), find all v (index j) that conflict.
    
    # We need a fast way to find vectors with specific support and signs.
    # Key: (frozenset(indices), signs_tuple) -> index
    # But signs are specific.
    # Let's create a dictionary for lookup.
    # Key: (idx1, idx2, idx3, idx4, s1, s2, s3, s4) -> index
    # This is unique.
    
    lookup = {}
    for i, (idxs, signs) in enumerate(type2_info):
        key = (tuple(sorted(idxs)), signs)
        lookup[key] = i
        
    # Actually, signs are tied to the order in idxs?
    # In generation: zip(idxs, signs). So sign[0] is for idxs[0].
    # So key should preserve order or we need to be careful.
    # Let's use (idxs, signs) directly as key, but idxs is a tuple from combinations (sorted).
    # So key = (idxs, signs) is unique.
    
    for i, (idxs, signs) in enumerate(type2_info):
        # idxs is a tuple of 4 sorted indices.
        # signs is a tuple of 4 signs.
        
        # For each subset of 3 indices from idxs
        for k in range(4):
            # Remove idxs[k]
            common_idx = idxs[:k] + idxs[k+1:]
            common_signs = signs[:k] + signs[k+1:]
            
            # The 4th index j in the conflicting vector v must be different from all in idxs.
            # j can be any of the D-4 = 7 indices not in idxs.
            # Let's find the set of indices not in idxs.
            # This is done inside the loop, might be slow.
            # Precompute complement?
            
            # For each j not in idxs:
            # The support of v is (common_idx + (j,)) sorted.
            # The signs of v on common_idx must match common_signs.
            # The sign of v on j can be 1 or -1.
            
            # We need to find the index j in the original coordinate system.
            # idxs is sorted.
            # Let's iterate j from 0 to D-1.
            for j in range(D):
                if j in idxs:
                    continue
                
                # Construct the support of v
                v_idxs = tuple(sorted(common_idx + (j,)))
                
                # Construct the signs of v
                # We need to map common_signs to the positions in v_idxs.
                # And assign a sign to j.
                
                # Let's create a dict for the known signs
                sign_map = {}
                for idx, s in zip(common_idx, common_signs):
                    sign_map[idx] = s
                
                for s_j in [1, -1]:
                    sign_map[j] = s_j
                    
                    # Construct the full signs tuple in the order of v_idxs
                    v_signs = tuple(sign_map[idx] for idx in v_idxs)
                    
                    key = (v_idxs, v_signs)
                    if key in lookup:
                        j_idx = lookup[key]
                        if j_idx != i:
                            adj[i].add(j_idx)
                            adj[j_idx].add(i)
                            
    # Now we have the conflict graph.
    # We want to find the maximum independent set.
    # Use greedy + local search.
    
    best_set = set()
    best_count = 0
    
    # Heuristic: Sort by degree
    degrees = [len(adj[i]) for i in range(N2)]
    
    # Try multiple random orderings
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
            
        if trial == 0:
            order = np.argsort(degrees, kind='stable')
        else:
            # Randomize order, but bias towards low degree?
            # Just random permutation
            order = rng.permutation(N2)
            
        selected = set()
        
        for i in order:
            # Check if i conflicts with any selected
            # Conflict if any neighbor is in selected
            is_conflict = False
            for n in adj[i]:
                if n in selected:
                    is_conflict = True
                    break
            if not is_conflict:
                selected.add(i)
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected.copy()
            
        # Local search: try to add more
        # Scan unselected, add if possible
        unselected = [i for i in range(N2) if i not in selected]
        # Sort unselected by degree?
        for i in unselected:
            if time.time() > deadline - 5:
                break
            is_conflict = False
            for n in adj[i]:
                if n in selected:
                    is_conflict = True
                    break
            if not is_conflict:
                selected.add(i)
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected.copy()

    # Combine with Type 1
    # Type 1 vectors are all compatible with Type 2
    # And Type 1 are mutually compatible
    final_indices = list(best_set)
    vectors = np.vstack([type1, type2[final_indices]])
    
    return vectors.astype(np.int64)
