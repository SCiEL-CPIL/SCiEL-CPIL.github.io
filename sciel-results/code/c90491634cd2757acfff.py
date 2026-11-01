import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to adjacency list for faster neighbor lookup
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)
    best_selected = None
    best_size = 0
    
    # Strategy 1: Greedy with min-degree
    alive = np.ones(N4, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = selected.copy()
    
    # Strategy 2: Multiple random greedy restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        alive = np.ones(N4, dtype=bool)
        selected = []
        order = rng.permutation(N4)
        for idx in order:
            if alive[idx]:
                selected.append(idx)
                to_remove = np.unique(np.concatenate(([idx], neighbors[idx])))
                alive[to_remove] = False
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    # Strategy 3: Local search with Kempe-chain swaps
    if best_selected is not None:
        selected_set = set(best_selected)
        selected_arr = np.array(best_selected)
        
        # Precompute conflict with selected
        conflict_with_selected = np.zeros(N4, dtype=bool)
        for idx in selected_arr:
            conflict_with_selected |= conflicts[idx]
        
        improved = True
        max_swap_rounds = 50
        round_count = 0
        while improved and round_count < max_swap_rounds and time.time() < deadline - 5:
            improved = False
            round_count += 1
            
            # Try 2-for-1 swaps: remove one selected, add two non-selected
            for i in range(len(selected_arr)):
                if time.time() > deadline - 5:
                    break
                remove_idx = selected_arr[i]
                remaining = np.setdiff1d(selected_arr, [remove_idx])
                
                # Find candidates: not in selected, not conflicting with remaining
                conflict_with_remaining = np.zeros(N4, dtype=bool)
                for idx in remaining:
                    conflict_with_remaining |= conflicts[idx]
                
                candidates = np.where(~conflict_with_remaining & ~np.isin(np.arange(N4), selected_arr))[0]
                
                # Try to find two non-conflicting candidates
                added = []
                cands_list = candidates.tolist()
                rng.shuffle(cands_list)
                for cand in cands_list:
                    if len(added) >= 2:
                        break
                    ok = True
                    for a in added:
                        if conflicts[cand, a]:
                            ok = False
                            break
                    if ok:
                        added.append(cand)
                
                if len(added) >= 2:
                    # Perform the swap
                    new_selected = list(selected_arr)
                    new_selected.pop(i)
                    new_selected.extend(added)
                    selected_arr = np.array(new_selected)
                    improved = True
                    best_size = len(selected_arr)
                    break
            
            # Try 3-for-2 swaps: remove two selected, add three non-selected
            if not improved and len(selected_arr) >= 2 and time.time() < deadline - 5:
                for i in range(len(selected_arr)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, len(selected_arr)):
                        if time.time() > deadline - 5:
                            break
                        remove_indices = [selected_arr[i], selected_arr[j]]
                        remaining = np.setdiff1d(selected_arr, remove_indices)
                        
                        conflict_with_remaining = np.zeros(N4, dtype=bool)
                        for idx in remaining:
                            conflict_with_remaining |= conflicts[idx]
                        
                        candidates = np.where(~conflict_with_remaining & ~np.isin(np.arange(N4), selected_arr))[0]
                        
                        # Try to find three non-conflicting candidates
                        added = []
                        cands_list = candidates.tolist()
                        rng.shuffle(cands_list)
                        for cand in cands_list:
                            if len(added) >= 3:
                                break
                            ok = True
                            for a in added:
                                if conflicts[cand, a]:
                                    ok = False
                                    break
                            if ok:
                                added.append(cand)
                        
                        if len(added) >= 3:
                            # Perform the swap
                            new_selected = list(selected_arr)
                            new_selected.pop(j)
                            new_selected.pop(i)
                            new_selected.extend(added)
                            selected_arr = np.array(new_selected)
                            improved = True
                            best_size = len(selected_arr)
                            break
                    if improved:
                        break
        
        best_selected = selected_arr.tolist()
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine V4 and V1
    # V1 vectors are all compatible with V4 (max dot = 2, not > 2)
    # V1 among themselves: can take both +2 and -2 for each position
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
