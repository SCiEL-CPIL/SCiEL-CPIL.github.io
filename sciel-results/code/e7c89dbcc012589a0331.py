import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Generate candidate vectors for weights 4, 5, 6
    # Condition for compatibility between u and v:
    # |u-v|^2 >= max(|u|^2, |v|^2)
    # |u|^2 + |v|^2 - 2 u.v >= max(|u|^2, |v|^2)
    # If |u|^2 == |v|^2 == N: u.v <= N/2
    # If |u|^2 < |v|^2: u.v <= |u|^2/2
    
    candidates = []
    norms_sq = []
    
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    # Weight 5
    for positions in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(5)
            
    # Weight 6
    for positions in itertools.combinations(range(D), 6):
        for signs in itertools.product([1, -1], repeat=6):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(6)
            
    V = np.array(candidates, dtype=np.int64)
    N = len(V)
    norms = np.array(norms_sq, dtype=np.int64)
    
    # Helper to check compatibility of a set of vectors
    # in_set: list of indices
    # Returns True if all pairs in in_set are compatible
    
    def is_compatible_set(indices):
        if len(indices) <= 1:
            return True
        # Get vectors
        vecs = V[indices]
        ns = norms[indices]
        # Compute pairwise dot products
        dots = vecs @ vecs.T
        # Condition: dots[i,j] <= min(ns[i], ns[j]) / 2
        # Since dots are integers, we can use: 2 * dots[i,j] <= min(ns[i], ns[j])
        min_ns = np.minimum.outer(ns, ns)
        # Check all off-diagonal
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if 2 * dots[i,j] > min_ns[i,j]:
                    return False
        return True

    # Greedy Initialization
    rng = np.random.default_rng(42)
    in_set = []
    available = np.arange(N)
    rng.shuffle(available)
    
    for idx in available:
        if time.time() > deadline - 20:
            break
        # Check if idx is compatible with all in in_set
        if not in_set:
            in_set.append(idx)
            continue
            
        # Fast check: compute dot products with all in set
        v_idx = V[idx]
        n_idx = norms[idx]
        
        compatible = True
        for j in in_set:
            v_j = V[j]
            n_j = norms[j]
            dot = np.dot(v_idx, v_j)
            if 2 * dot > min(n_idx, n_j):
                compatible = False
                break
        
        if compatible:
            in_set.append(idx)

    # Local Search
    # We will try to swap vectors in the set with vectors outside
    # to potentially increase the size or improve "slack"
    
    # To make local search efficient, we pre-compute dot products of all candidates with current set
    # But the set changes.
    
    def get_compatible_outside_indices(in_set_list):
        """
        Returns list of indices not in in_set_list that are compatible with all vectors in in_set_list.
        """
        if not in_set_list:
            return list(range(N))
            
        set_vecs = V[in_set_list]
        set_norms = norms[in_set_list]
        K = len(in_set_list)
        
        # Compute dot products of all N vectors with K set vectors
        # Dots shape: (N, K)
        dots_all = V @ set_vecs.T
        
        # Min norms for each pair (N, K)
        # min_norms[i, j] = min(norms[i], norms[in_set_list[j]])
        # We can broadcast
        min_norms = np.minimum.outer(norms, set_norms) # Shape (N, K)
        
        # Condition: 2 * dots_all[i, j] <= min_norms[i, j] for all j
        # valid[i] = all(2 * dots_all[i, :] <= min_norms[i, :])
        valid = np.all(2 * dots_all <= min_norms, axis=1)
        
        # Exclude indices already in set
        in_set_set = set(in_set_list)
        for idx in in_set_list:
            valid[idx] = False
            
        return np.where(valid)[0]

    # Main Local Search Loop
    # We iterate until no improvement or time limit
    # Improvement: adding a vector if possible, or swapping to allow future additions?
    # Since we want max size, we primarily look for additions.
    # If no additions possible, we try swaps to see if we can unblock additions.
    
    current_set = in_set.copy()
    best_set = current_set.copy()
    best_size = len(current_set)
    
    # Pre-compute some things for speed?
    # The `get_compatible_outside_indices` is O(N*K*D).
    # N=50000, K=50, D=11. ~27M ops.
    # This is fast enough to run many times.
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # 1. Try to add any compatible vector
        addable = get_compatible_outside_indices(current_set)
        
        if len(addable) > 0:
            # Add one (randomly or best heuristic)
            # Heuristic: pick one that has the most "slack" or allows most future additions?
            # For now, pick randomly to explore
            v_add = addable[rng.integers(len(addable))]
            current_set.append(v_add)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            continue # Restart loop to check for more additions
        
        # 2. If no additions, try swaps
        # Try swapping each vector in current_set with a compatible outside vector
        # If swap increases size? No, swap keeps size same.
        # But it might allow an addition afterwards.
        
        swapped = False
        for i, v_out_idx in enumerate(current_set):
            if time.time() > deadline - 5:
                break
                
            # Temporarily remove v_out_idx
            temp_set = current_set[:i] + current_set[i+1:]
            
            # Find addable vectors for temp_set
            # Note: v_out_idx is now "outside"
            addable_temp = get_compatible_outside_indices(temp_set)
            
            # We want to swap v_out_idx with some v_in from addable_temp
            # Ideally, we pick v_in such that it allows more future additions.
            # Simple heuristic: just pick one and see if we can add another.
            
            if len(addable_temp) > 0:
                # Pick a candidate
                v_in = addable_temp[rng.integers(len(addable_temp))]
                
                # Perform swap
                new_set = temp_set + [v_in]
                
                # Check if we can add any more vectors to new_set
                addable_new = get_compatible_outside_indices(new_set)
                
                if len(addable_new) > 0:
                    # Great, this swap allowed an addition
                    # Add one
                    v_add = addable_new[rng.integers(len(addable_new))]
                    new_set.append(v_add)
                    
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = new_set.copy()
                    
                    current_set = new_set
                    swapped = True
                    break # Restart main loop
                else:
                    # Swap didn't immediately allow addition, but might be better structure?
                    # For now, ignore swaps that don't lead to immediate size increase
                    # to keep search focused on size.
                    pass
        
        if not swapped:
            # No improvement possible with single swaps leading to additions
            # Try 2-swaps? Or restart with different random seed?
            # For now, break if stuck
            # Or try a different random addition order?
            
            # Let's try a "kick" move: remove 2 random, add 2 random compatible
            if len(current_set) >= 2:
                # Pick 2 to remove
                rem_idx = rng.choice(len(current_set), size=2, replace=False)
                rem_idx = sorted(rem_idx, reverse=True)
                v1 = current_set.pop(rem_idx[0])
                v2 = current_set.pop(rem_idx[1])
                
                temp_set = current_set.copy()
                
                # Find 2 addable
                addable_temp = get_compatible_outside_indices(temp_set)
                
                if len(addable_temp) >= 2:
                    # Pick 2
                    add_idx = rng.choice(len(addable_temp), size=2, replace=False)
                    v1_new = addable_temp[add_idx[0]]
                    v2_new = addable_temp[add_idx[1]]
                    
                    new_set = temp_set + [v1_new, v2_new]
                    
                    # Check for additions
                    addable_new = get_compatible_outside_indices(new_set)
                    if len(addable_new) > 0:
                         v_add = addable_new[rng.integers(len(addable_new))]
                         new_set.append(v_add)
                         
                         if len(new_set) > best_size:
                             best_size = len(new_set)
                             best_set = new_set.copy()
                         
                         current_set = new_set
                         swapped = True
                         continue
                 
                # Restore if no improvement
                # Note: current_set was modified in place.
                # We need to restore v1, v2
                current_set.append(v1)
                current_set.append(v2)
                # Order might be wrong, but set is unordered for logic.
                # However, for consistent indexing, let's just revert to best if stuck.
                
            # If still stuck, break
            break

    # Final Polish: Try to add any remaining vectors to best_set
    # This ensures we return a maximal set
    final_set = best_set.copy()
    addable_final = get_compatible_outside_indices(final_set)
    while len(addable_final) > 0:
        v_add = addable_final[rng.integers(len(addable_final))]
        final_set.append(v_add)
        addable_final = get_compatible_outside_indices(final_set)
        
    # Return the vectors
    if not final_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = V[final_set]
    return result

# EVOLVE-BLOCK-END
