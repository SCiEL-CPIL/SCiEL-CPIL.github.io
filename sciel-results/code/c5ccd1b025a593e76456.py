import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    d = 11
    
    # 1. Generate Candidates
    candidates = []
    
    # 1-norm
    for i in range(d):
        for val in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = val
            candidates.append(v)
            
    # 2-norm
    for i in range(d):
        for j in range(i+1, d):
            for a in [-1, 1]:
                for b in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = a
                    v[j] = b
                    candidates.append(v)
                    
    # 3-norm
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for a in [-1, 1]:
                    for b in [-1, 1]:
                        for c in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = a
                            v[j] = b
                            v[k] = c
                            candidates.append(v)
                            
    # 4-norm
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    for a in [-1, 1]:
                        for b in [-1, 1]:
                            for c in [-1, 1]:
                                for e in [-1, 1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = a
                                    v[j] = b
                                    v[k] = c
                                    v[l] = e
                                    candidates.append(v)
                                    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # 2. Precompute Norms and Distances
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Pairwise squared distances
    dots = candidates @ candidates.T
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    dists_sq = (dists_sq + dists_sq.T) / 2.0
    np.maximum(dists_sq, 0, out=dists_sq)
    
    max_norms_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    # valid_matrix: True if dist_sq >= max_norm_sq
    valid_matrix = dists_sq >= max_norms_sq
    np.fill_diagonal(valid_matrix, True)
    
    # 3. Local Search
    # Greedy Init
    idx_order = np.random.permutation(n_cand)
    selected = []
    for idx in idx_order:
        if time.time() > deadline - 10.0:
            break
        if not selected:
            selected.append(idx)
        elif np.all(valid_matrix[idx, selected]):
            selected.append(idx)
            
    if not selected:
        selected = [0]
        
    current_set = set(selected)
    best_set = list(current_set)
    best_size = len(current_set)
    
    lam = 10.0
    
    def get_penalty(indices):
        if len(indices) < 2:
            return 0.0
        idx_arr = np.array(indices)
        sub_dists = dists_sq[np.ix_(idx_arr, idx_arr)]
        sub_max_norms = max_norms_sq[np.ix_(idx_arr, idx_arr)]
        n = len(idx_arr)
        iu = np.triu_indices(n, k=1)
        viol = np.maximum(0, sub_max_norms[iu] - sub_dists[iu])
        return np.sum(viol * viol)
        
    # Search Loop
    while time.time() < deadline - 5.0:
        curr_pen = get_penalty(list(current_set))
        improved = False
        
        # Try Add
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            cand_idx = np.random.randint(n_cand)
            if cand_idx in current_set:
                continue
            
            new_set = list(current_set) + [cand_idx]
            new_pen = get_penalty(new_set)
            
            # Score: -size + lam * pen
            if -len(new_set) + lam * new_pen < -len(current_set) + lam * curr_pen:
                current_set = set(new_set)
                improved = True
                break
                
        # Try Kick (Remove + Refill)
        if not improved and len(current_set) > 1:
            rem_idx = random.choice(list(current_set))
            reduced_set = list(current_set)
            reduced_set.remove(rem_idx)
            
            # Refill: try to add candidates to reduced_set
            # We want to maximize size, so we prefer valid additions.
            # If we can add a valid one, we restore size.
            # If we can add an invalid one that leads to better future, we might take it?
            # For simplicity, try to add any candidate that reduces penalty or is valid.
            
            added = False
            # Try random candidates
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                cand_idx = np.random.randint(n_cand)
                if cand_idx in reduced_set:
                    continue
                
                test_set = reduced_set + [cand_idx]
                test_pen = get_penalty(test_set)
                
                # Accept if it restores size and is valid (pen=0)
                # Or if it reduces penalty significantly
                if test_pen == 0:
                    current_set = set(test_set)
                    improved = True
                    added = True
                    break
                elif test_pen < curr_pen:
                    # Even if size is same (reduced+1 = original size), if penalty is lower, it's a better state
                    current_set = set(test_set)
                    improved = True
                    added = True
                    break
            
            # If we couldn't add anything good, maybe just keep the reduced set?
            # No, that loses size. 
            # Let's try to add ANY valid candidate to the reduced set.
            if not added:
                for _ in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    cand_idx = np.random.randint(n_cand)
                    if cand_idx in reduced_set:
                        continue
                    if valid_matrix[cand_idx, reduced_set].all():
                        current_set = set(reduced_set + [cand_idx])
                        improved = True
                        break
                        
        # Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    # Final Pruning: Ensure best_set is valid
    # If best_set is not valid, we need to remove vectors until it is.
    # Simple iterative removal of the most violating vector.
    final_indices = list(best_set)
    
    def is_valid(indices):
        if len(indices) < 2:
            return True
        idx_arr = np.array(indices)
        sub_dists = dists_sq[np.ix_(idx_arr, idx_arr)]
        sub_max_norms = max_norms_sq[np.ix_(idx_arr, idx_arr)]
        n = len(idx_arr)
        iu = np.triu_indices(n, k=1)
        viol = np.maximum(0, sub_max_norms[iu] - sub_dists[iu])
        return np.sum(viol) == 0.0
        
    # Prune
    # We assume best_set was tracked to be "good", but might have small violations due to search dynamics.
    # If we only accepted moves that reduced score, and score includes penalty, 
    # it's possible we ended up in a state with penalty > 0 if -size gain was huge.
    # So we must prune.
    
    while not is_valid(final_indices) and len(final_indices) > 1:
        # Find the pair with largest violation
        idx_arr = np.array(final_indices)
        sub_dists = dists_sq[np.ix_(idx_arr, idx_arr)]
        sub_max_norms = max_norms_sq[np.ix_(idx_arr, idx_arr)]
        n = len(idx_arr)
        iu = np.triu_indices(n, k=1)
        viol = np.maximum(0, sub_max_norms[iu] - sub_dists[iu])
        
        if np.sum(viol) == 0:
            break
            
        # Find the index with highest sum of violations
        viol_sum = np.zeros(n)
        # Map iu indices back to 0..n-1
        # iu[0] is row, iu[1] is col
        for i, j in zip(iu[0], iu[1]):
            if viol[i] > 0: # viol is 1D array corresponding to iu pairs
                pass # This logic is tricky with 1D viol array.
                
        # Easier: just remove a random vector and recheck? 
        # Or remove the one involved in the max violation.
        max_viol_idx = np.argmax(viol)
        # Get the pair
        i, j = iu[0][max_viol_idx], iu[1][max_viol_idx]
        # Remove the one with larger norm? Or random?
        # Remove random one from the pair to break tie
        if random.random() < 0.5:
            final_indices.pop(i)
        else:
            final_indices.pop(j)
            
    return np.array(final_indices, dtype=np.int64)
