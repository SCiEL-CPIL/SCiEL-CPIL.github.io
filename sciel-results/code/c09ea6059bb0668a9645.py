import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Generate candidates
    # Type 1: One +/- 2 (22 vectors)
    # Type 2: Four +/- 1 (5280 vectors)
    
    # We will solve for Type 2, then add Type 1.
    
    # Generate Type 2 vectors
    v2_list = []
    v2_indices = [] # (i1, i2, i3, i4)
    v2_signs = []   # (s1, s2, s3, s4)
    
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v2_list.append((idxs, signs))
            
    N2 = len(v2_list)
    
    # Build conflict graph for Type 2
    # Conflict if dot product >= 3
    # Dot product is sum of products of signs at common indices.
    # Max dot 4 (identical), 3 (3 common same sign, 1 diff index), 2 (2 common same sign, or 3 common with 1 diff sign? No, 3 common with 1 diff sign is dot 2).
    
    # Let's precompute conflicts
    # conflict[i] is a set of j such that v_i conflicts with v_j
    
    # Optimization: Group by support?
    # Two vectors conflict if their supports overlap in 3 elements with matching signs.
    
    # Let's just compute pairwise for now if N2 is small enough?
    # 5280^2 is 28M. Too slow in Python loops.
    # Use bitsets or structured lookup.
    
    # Lookup table:
    # Key: (sorted_indices, signs_tuple) -> index in v2_list
    # But signs matter for conflict.
    
    # Alternative:
    # For each vector v, find all w that conflict with it.
    # w must share 3 indices with v.
    # Let v have indices i1, i2, i3, i4 and signs s1, s2, s3, s4.
    # Possible conflicts:
    # 1. Identical: (i1,i2,i3,i4, s1,s2,s3,s4)
    # 2. Share 3 indices:
    #    - Drop i1: Keep i2,i3,i4 with signs s2,s3,s4. New index j != i1,i2,i3,i4. Sign t.
    #    - Drop i2: Keep i1,i3,i4 with signs s1,s3,s4. New index j. Sign t.
    #    - Drop i3: Keep i1,i2,i4 with signs s1,s2,s4. New index j. Sign t.
    #    - Drop i4: Keep i1,i2,i3 with signs s1,s2,s3. New index j. Sign t.
    
    # We need a fast way to find the index of a vector given its indices and signs.
    
    # Create a dictionary mapping (tuple(sorted_indices), tuple(signs_in_sorted_order)) to index?
    # No, signs are tied to specific indices.
    # Key: (i1, i2, i3, i4, s1, s2, s3, s4) where i1<i2<i3<i4.
    
    index_map = {}
    for k, (idxs, signs) in enumerate(v2_list):
        # idxs is already sorted from combinations
        key = (idxs[0], idxs[1], idxs[2], idxs[3], signs[0], signs[1], signs[2], signs[3])
        index_map[key] = k
        
    conflict_list = [[] for _ in range(N2)]
    
    for k, (idxs, signs) in enumerate(v2_list):
        i1, i2, i3, i4 = idxs
        s1, s2, s3, s4 = signs
        
        # 1. Identical
        conflict_list[k].append(k)
        
        # 2. Share 3 indices
        # Case: Drop i1. Keep i2, i3, i4 with s2, s3, s4.
        # New index j can be any of 0..10 except i1,i2,i3,i4.
        # Sign t can be 1 or -1.
        for j in range(D):
            if j in (i1, i2, i3, i4): continue
            for t in [1, -1]:
                # Construct key for new vector
                # Indices: sorted(i2, i3, i4, j)
                # Signs: corresponding to sorted indices
                new_idxs = sorted([i2, i3, i4, j])
                # Need to map signs to new sorted indices
                # Original signs for i2,i3,i4 are s2,s3,s4. Sign for j is t.
                sign_map = {i2: s2, i3: s3, i4: s4, j: t}
                new_signs = tuple(sign_map[x] for x in new_idxs)
                key = (new_idxs[0], new_idxs[1], new_idxs[2], new_idxs[3], new_signs[0], new_signs[1], new_signs[2], new_signs[3])
                if key in index_map:
                    conflict_list[k].append(index_map[key])
                    
        # Case: Drop i2. Keep i1, i3, i4 with s1, s3, s4.
        for j in range(D):
            if j in (i1, i2, i3, i4): continue
            for t in [1, -1]:
                new_idxs = sorted([i1, i3, i4, j])
                sign_map = {i1: s1, i3: s3, i4: s4, j: t}
                new_signs = tuple(sign_map[x] for x in new_idxs)
                key = (new_idxs[0], new_idxs[1], new_idxs[2], new_idxs[3], new_signs[0], new_signs[1], new_signs[2], new_signs[3])
                if key in index_map:
                    conflict_list[k].append(index_map[key])
                    
        # Case: Drop i3. Keep i1, i2, i4 with s1, s2, s4.
        for j in range(D):
            if j in (i1, i2, i3, i4): continue
            for t in [1, -1]:
                new_idxs = sorted([i1, i2, i4, j])
                sign_map = {i1: s1, i2: s2, i4: s4, j: t}
                new_signs = tuple(sign_map[x] for x in new_idxs)
                key = (new_idxs[0], new_idxs[1], new_idxs[2], new_idxs[3], new_signs[0], new_signs[1], new_signs[2], new_signs[3])
                if key in index_map:
                    conflict_list[k].append(index_map[key])
                    
        # Case: Drop i4. Keep i1, i2, i3 with s1, s2, s3.
        for j in range(D):
            if j in (i1, i2, i3, i4): continue
            for t in [1, -1]:
                new_idxs = sorted([i1, i2, i3, j])
                sign_map = {i1: s1, i2: s2, i3: s3, j: t}
                new_signs = tuple(sign_map[x] for x in new_idxs)
                key = (new_idxs[0], new_idxs[1], new_idxs[2], new_idxs[3], new_signs[0], new_signs[1], new_signs[2], new_signs[3])
                if key in index_map:
                    conflict_list[k].append(index_map[key])
                    
    # Remove duplicates in conflict lists?
    # The generation might add the same conflict multiple times?
    # No, each case is distinct (dropping different index).
    # But we should use sets to be safe and unique.
    for k in range(N2):
        conflict_list[k] = list(set(conflict_list[k]))
        
    # Convert to numpy arrays for fast degree updates?
    # conflict_list is list of lists.
    # We need fast "add 1 to degree of neighbors".
    
    # Greedy Initialization
    rng = np.random.default_rng(42)
    
    best_set = []
    best_size = 0
    
    # Try multiple random orderings for greedy
    for run in range(20):
        if time.time() > deadline - 10:
            break
            
        order = rng.permutation(N2)
        in_set = np.zeros(N2, dtype=bool)
        degree_in_set = np.zeros(N2, dtype=np.int32)
        current_set = []
        
        for idx in order:
            if degree_in_set[idx] == 0:
                # Add idx
                in_set[idx] = True
                current_set.append(idx)
                # Update degrees of neighbors
                for nb in conflict_list[idx]:
                    degree_in_set[nb] += 1
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Local Search on the best set found
    if time.time() < deadline - 5 and best_set:
        in_set = np.zeros(N2, dtype=bool)
        for idx in best_set:
            in_set[idx] = True
            
        degree_in_set = np.zeros(N2, dtype=np.int32)
        # Recalculate degrees for the current set
        for idx in best_set:
            for nb in conflict_list[idx]:
                degree_in_set[nb] += 1
                
        # We want to find addable vertices
        # addable = ~in_set & (degree_in_set == 0)
        
        # Iterative improvement
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Find addable
            addable = np.where((~in_set) & (degree_in_set == 0))[0]
            
            if len(addable) > 0:
                # Add one (or more if we want, but one at a time is safer for tracking)
                # Pick one with min degree in the "remaining" graph?
                # Or just the first one.
                idx = addable[0]
                in_set[idx] = True
                best_set.append(idx)
                for nb in conflict_list[idx]:
                    degree_in_set[nb] += 1
                improved = True
                best_size = len(best_set)
            else:
                # No addable vertices. Try a swap.
                # Pick a random vertex from best_set to remove
                if len(best_set) == 0:
                    break
                    
                # Remove a vertex that might unlock many others?
                # Heuristic: remove vertex with highest degree in the conflict graph?
                # Or random.
                # Let's try random first, then maybe heuristic.
                
                # To avoid infinite loops of swapping same vertices, we can track recent moves or just rely on time limit.
                
                u = best_set[rng.integers(len(best_set))]
                
                # Remove u
                in_set[u] = False
                best_set.remove(u)
                
                for nb in conflict_list[u]:
                    degree_in_set[nb] -= 1
                    
                # Check for addable
                addable = np.where((~in_set) & (degree_in_set == 0))[0]
                
                if len(addable) > 0:
                    # Add as many as possible?
                    # If we add one, it might block others.
                    # Let's add one and see if we improved size.
                    # If we added 1, size is same (removed 1, added 1).
                    # If we can add 2, size increases.
                    
                    # Try to add the one that unlocks the most?
                    # Or just add one.
                    idx = addable[0]
                    in_set[idx] = True
                    best_set.append(idx)
                    for nb in conflict_list[idx]:
                        degree_in_set[nb] += 1
                    
                    # If size didn't increase, we might be stuck.
                    # To escape local minima, we could allow "worse" moves (not possible here as we only add if degree 0).
                    # The "swap" itself is the move.
                    # If we removed 1 and added 1, we are at same size.
                    # If we removed 1 and added 0, we should revert.
                    # Here we added 1, so size is same as before swap.
                    # This allows exploration.
                    improved = True # We made a move
                else:
                    # Could not add anything. Revert.
                    in_set[u] = True
                    best_set.append(u)
                    for nb in conflict_list[u]:
                        degree_in_set[nb] += 1
                    # improved = False (implicit)

    # Construct final vectors
    # Add V1 (22 vectors)
    v1_vectors = []
    for i in range(D):
        v1_vectors.append([0]*D)
        v1_vectors[-1][i] = 2
        v1_vectors.append([0]*D)
        v1_vectors[-1][i] = -2
        
    # Add V2 from best_set
    v2_vectors = []
    for idx in best_set:
        idxs, signs = v2_list[idx]
        v = [0]*D
        for i, s in zip(idxs, signs):
            v[i] = s
        v2_vectors.append(v)
        
    all_vectors = v1_vectors + v2_vectors
    result = np.array(all_vectors, dtype=np.int64)
    
    # Verify validity
    n = len(result)
    if n == 0:
         return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
         
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    # Optimization: only check pairs
    for i in range(n):
        for j in range(i+1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to just V1
        return np.array(v1_vectors, dtype=np.int64)
        
    return result
