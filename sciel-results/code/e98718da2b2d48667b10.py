import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
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
    
    # Precompute conflict matrix: conflict[i,j] = True if M[i,j] > 2
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Vectorized greedy select
    def greedy_select(order, conflict_mat, N):
        selected_orig = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if count == 0:
                selected_orig[idx] = True
                count += 1
            else:
                if not np.any(conflict_mat[idx] & selected_orig):
                    selected_orig[idx] = True
                    count += 1
        return selected_orig, count
    
    # Strategy A: Many random orderings
    n_trials = 2000
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_mask, count = greedy_select(order, conflict, N)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Least conflicts first
    if time.time() < deadline - 30:
        conflict_counts = np.sum(conflict, axis=1)
        order_b = np.argsort(conflict_counts, kind='stable')
        selected_mask, count = greedy_select(order_b, conflict, N)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy C: Most compatible first
    if time.time() < deadline - 20:
        compat_counts = N - 1 - np.sum(conflict, axis=1)
        order_c = np.argsort(-compat_counts, kind='stable')
        selected_mask, count = greedy_select(order_c, conflict, N)
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy D: Local search
    if best_set is not None and time.time() < deadline - 5:
        # Phase 1: Add any compatible unselected vectors
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            if not np.any(conflict[idx] & best_set):
                best_set[idx] = True
                best_count += 1
        
        # Phase 2: Local search - remove some, re-add greedily
        for ls_iter in range(30):
            if time.time() > deadline - 3:
                break
            
            selected_indices = np.where(best_set)[0]
            if len(selected_indices) < 5:
                break
            
            # Try removing 2-3 random selected and re-adding greedily
            n_remove = min(3, len(selected_indices))
            remove_candidates = rng.choice(len(selected_indices), size=n_remove, replace=False)
            remove_set = selected_indices[remove_candidates]
            
            test_set = best_set.copy()
            for r in remove_set:
                test_set[r] = False
            test_count = best_count - n_remove
            
            # Greedily add from unselected, prioritizing low-conflict
            unselected_test = np.where(~test_set)[0]
            if len(unselected_test) > 0:
                conf_counts = np.array([np.sum(conflict[u] & test_set) for u in unselected_test])
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
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
