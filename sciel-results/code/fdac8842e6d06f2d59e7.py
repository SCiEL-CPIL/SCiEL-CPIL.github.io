import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: two vectors conflict if their distance < 2, i.e., |v_i - v_j|^2 < 4
    # |v_i - v_j|^2 = |v_i|^2 + |v_j|^2 - 2*v_i·v_j = 4 + 4 - 2*M[i,j] = 8 - 2*M[i,j]
    # Need distance >= 2, so |v_i - v_j|^2 >= 4, so 8 - 2*M[i,j] >= 4, M[i,j] <= 2
    # Conflict if M[i,j] > 2
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute conflict counts for each vertex
    conflict_counts = np.sum(conflict, axis=1)
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy A: Get initial solution via multiple greedy passes with different orderings
    n_trials = 2000
    for trial in range(n_trials):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            # Order by ascending conflict count (least constrained first)
            order = np.argsort(conflict_counts)
        elif trial == 1:
            order = np.arange(N)
        elif trial == 2:
            order = np.argsort(-conflict_counts)  # most constrained first
        else:
            # Random permutation
            order = rng.permutation(N)
        
        # Vectorized greedy: iterate in order, add if no conflict with selected
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if not np.any(conflict[idx] & selected_mask):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Local search with vectorized operations
    if best_set is not None and time.time() < deadline - 10:
        for ls_iter in range(200):
            if time.time() > deadline - 5:
                break
            
            selected_indices = np.where(best_set)[0]
            if len(selected_indices) < 5:
                break
            
            improved = False
            
            # Try removing 1-4 random selected and re-adding greedily
            n_remove = min(rng.integers(1, 5), len(selected_indices))
            remove_candidates = rng.choice(len(selected_indices), size=n_remove, replace=False)
            remove_set = selected_indices[remove_candidates]
            
            test_set = best_set.copy()
            for r in remove_set:
                test_set[r] = False
            test_count = best_count - n_remove
            
            # Get unselected vertices and compute their conflict with test_set
            unselected_mask = ~test_set
            unselected_test = np.where(unselected_mask)[0]
            
            if len(unselected_test) > 0:
                # Vectorized: count conflicts for each unselected vertex with test_set
                # conflict[u] & test_set gives a boolean array; sum gives count
                conf_counts = np.array([np.sum(conflict[u] & test_set) for u in unselected_test])
                
                # Sort by ascending conflict count (least conflicting first)
                order_add = unselected_test[np.argsort(conf_counts)]
                
                for add_idx in order_add:
                    if time.time() > deadline - 3:
                        break
                    if not np.any(conflict[add_idx] & test_set):
                        test_set[add_idx] = True
                        test_count += 1
            
            if test_count > best_count:
                best_set = test_set
                best_count = test_count
                improved = True
            
            # Also try: remove 1 vertex, see if we can add 2+
            if not improved and len(selected_indices) > 3:
                for r in rng.choice(len(selected_indices), size=min(10, len(selected_indices)), replace=False):
                    if time.time() > deadline - 3:
                        break
                    rem_idx = selected_indices[r]
                    test_set2 = best_set.copy()
                    test_set2[rem_idx] = False
                    test_count2 = best_count - 1
                    
                    unselected_test2 = np.where(~test_set2)[0]
                    if len(unselected_test2) > 0:
                        conf_counts2 = np.array([np.sum(conflict[u] & test_set2) for u in unselected_test2])
                        order_add2 = unselected_test2[np.argsort(conf_counts2)]
                        
                        for add_idx in order_add2:
                            if time.time() > deadline - 3:
                                break
                            if not np.any(conflict[add_idx] & test_set2):
                                test_set2[add_idx] = True
                                test_count2 += 1
                    
                    if test_count2 > best_count:
                        best_set = test_set2
                        best_count = test_count2
                        break
    
    # Strategy C: Final sweep - add any remaining compatible vectors
    if best_set is not None and time.time() < deadline - 1:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 0.5:
                break
            if not np.any(conflict[idx] & best_set):
                best_set[idx] = True
                best_count += 1
    
    # Strategy D: Try a few more greedy passes on the remaining unselected
    if best_set is not None and time.time() < deadline - 2:
        unselected = np.where(~best_set)[0]
        if len(unselected) > 0:
            for _ in range(100):
                if time.time() > deadline - 1:
                    break
                order = unselected[rng.permutation(len(unselected))]
                added = 0
                for idx in order:
                    if not np.any(conflict[idx] & best_set):
                        best_set[idx] = True
                        best_count += 1
                        added += 1
                if added == 0:
                    break
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
