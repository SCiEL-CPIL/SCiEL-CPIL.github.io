import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate weight-4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate weight-1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute dot products and conflicts
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    best_selected = None
    best_size = 0
    
    # Multiple restarts of greedy MIS + local search
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Greedy MIS
        alive = np.ones(N4, dtype=bool)
        selected = []
        
        if restart % 2 == 0:
            # Min-degree greedy
            degrees = conflicts.sum(axis=1).astype(np.int32)
            while np.any(alive):
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0: break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                chosen = cands[rng.integers(len(cands))]
                selected.append(chosen)
                neighbors = np.where(conflicts[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        else:
            # Random greedy
            order = rng.permutation(N4)
            for idx in order:
                if alive[idx]:
                    selected.append(idx)
                    neighbors = np.where(conflicts[idx])[0]
                    to_remove = np.unique(np.concatenate(([idx], neighbors)))
                    alive[to_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            
        # Local search: 1-removal, k-addition
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_sel = len(selected)
            # Shuffle order of removals to avoid bias
            order = rng.permutation(n_sel)
            for i in order:
                if time.time() > deadline - 5: break
                remove_idx = selected[i]
                # Compute compatibility with selected minus remove_idx
                # selected_mask is boolean array of size N4
                in_selected = np.zeros(N4, dtype=bool)
                in_selected[selected] = True
                in_selected[remove_idx] = False
                
                # Conflict with remaining selected
                # sum of conflicts of all in_selected (excluding remove_idx)
                # We can precompute conflict sums?
                # conflicts is N4xN4 bool.
                # conflict_with_rem = np.any(conflicts[in_selected], axis=0)
                # This is O(N4 * N_sel) per step. N_sel ~ 100. N4 ~ 5000. 500k ops. Fast.
                
                conflict_with_rem = np.any(conflicts[in_selected], axis=0)
                
                # Candidates: not in selected (including remove_idx? no, remove_idx is in selected originally, but we are testing if we can add others. We removed it. Can we add it back? No, we are replacing it. But we can add vectors that were previously conflicting with remove_idx.
                # Candidates must not be in the REMAINING selected set.
                # They CAN be the removed vector? If we remove it and add it back, size doesn't change. We want size increase.
                # So candidates = ~conflict_with_rem & ~in_selected
                # Note: in_selected[remove_idx] is False now. So remove_idx is a candidate.
                # If we add remove_idx back, we are back to start.
                # We need to add at least 2 vectors to increase size by 1 (remove 1, add 2).
                
                candidates_mask = ~conflict_with_rem & ~in_selected
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) < 2:
                    continue
                
                # Greedy MIS on candidates
                # We want to pick as many as possible.
                # Simple greedy: pick node with min degree in candidate subgraph.
                cand_list = candidates.tolist()
                cand_set = set(cand_list)
                temp_selected = []
                remaining_cands = set(cand_list)
                
                # To speed up, we can just pick 2 if possible?
                # If we pick 2, size increases by 1.
                # Let's try to pick 2 compatible vectors.
                # If we find 2, we swap.
                # If we find more, even better.
                
                # Find 2 compatible vectors in candidates
                found = False
                # Sort by degree in candidate subgraph?
                # Just iterate
                for i_c in range(len(cand_list)):
                    if found: break
                    v1 = cand_list[i_c]
                    # Find v2 in cand_list that is compatible with v1
                    # and v1, v2 compatible with remaining selected (already ensured by candidates_mask)
                    # and v1 != v2
                    # Check conflicts[v1, candidates]
                    compat_v1 = np.where(~conflicts[v1] & candidates_mask)[0]
                    # Remove v1 itself
                    compat_v1 = compat_v1[compat_v1 != v1]
                    if len(compat_v1) > 0:
                        v2 = compat_v1[0]
                        # Swap
                        new_selected = [x for x in selected if x != remove_idx]
                        new_selected.append(v1)
                        new_selected.append(v2)
                        selected = new_selected
                        improved = True
                        found = True
                        break
                        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            
    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
