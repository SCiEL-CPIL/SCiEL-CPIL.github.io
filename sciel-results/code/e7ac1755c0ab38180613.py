import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety

    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        import itertools
        if n >= 4:
            # Simple product construction for general case
            # This is a heuristic, not optimal
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Just return a random set or full space if parameters are weird
            # For this task, we assume q=3, n=8, r=1
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    # Pre-compute neighbors
    # neighbors[i] contains indices of all words within distance 1 of i
    neighbors = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return w
        
    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Pre-compute neighbors
    for i in range(total_words):
        w = idx_to_word(i)
        neigh_list = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neigh_list.append(word_to_idx(w))
            w[pos] = orig_val
        neighbors[i] = neigh_list

    # 1. Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected_indices = []
    
    while True:
        if time.time() > deadline - 10.0: # Reserve time for refinement
            break
            
        counts = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if not uncovered.any():
            break

    # 2. Local Search / Refinement
    # We want to minimize len(selected_indices)
    # Track coverage counts for all words
    # count[w] = number of selected codewords that cover w
    
    # Initialize counts
    # Efficient way: 
    # counts = np.bincount([j for i in selected_indices for j in neighbors[i]], minlength=total_words)
    # But this is slow in Python. 
    # Use numpy:
    # Flatten neighbors of selected indices
    # selected_neighbors = neighbors[selected_indices]
    # counts = np.bincount(selected_neighbors.ravel(), minlength=total_words)
    
    if not selected_indices:
        return [idx_to_word(i) for i in selected_indices]

    selected_set = set(selected_indices)
    selected_array = np.array(selected_indices)
    
    # Calculate initial counts
    # neighbors[selected_array] is shape (M, 17)
    # We want sum of 1s for each column index across rows
    # This is equivalent to bincount of the flattened array
    flat_neigh = neighbors[selected_array].ravel()
    counts = np.bincount(flat_neigh, minlength=total_words)
    
    # Refinement Loop
    # We try to remove codewords if possible.
    # If not, we try to swap.
    
    # Helper to check if a codeword can be removed
    # A codeword c can be removed if for all w in N(c), counts[w] > 1
    
    def can_remove(c_idx):
        # If any neighbor has count 1, we cannot remove
        # counts[neighbors[c_idx]] will have 1s if critical
        # We check if min is 1
        return np.min(counts[neighbors[c_idx]]) > 1

    # Helper to perform removal
    def remove_codeword(c_idx):
        nonlocal selected_set, selected_array, counts
        counts[neighbors[c_idx]] -= 1
        selected_set.remove(c_idx)
        selected_array = np.array(list(selected_set))
        # Note: selected_array is rebuilt, which is O(M). 
        # For M~400, this is fine.

    # Helper to perform swap
    def swap_codeword(old_idx, new_idx):
        nonlocal selected_set, selected_array, counts
        counts[neighbors[old_idx]] -= 1
        counts[neighbors[new_idx]] += 1
        selected_set.remove(old_idx)
        selected_set.add(new_idx)
        selected_array = np.array(list(selected_set))

    # Main refinement loop
    # We iterate until no improvements or time runs out
    # To avoid getting stuck, we can use a simple random restart or just iterate through codewords
    
    # Optimization: Keep a list of "critical" words for each codeword?
    # That's expensive to update.
    # Instead, just check can_remove for each codeword.
    
    # Let's try a pass of removals first
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate in random order to avoid bias
        current_list = list(selected_set)
        np.random.shuffle(current_list)
        for c in current_list:
            if c not in selected_set:
                continue
            if can_remove(c):
                remove_codeword(c)
                improved = True
                # After removal, counts changed, so previous checks are invalid
                # But we are iterating over a snapshot. 
                # We can break and restart the loop for efficiency
                break
        
    # Now try swaps
    # Strategy: Find a codeword with the smallest number of "critical" neighbors
    # and try to swap it with a codeword that covers those critical neighbors.
    
    # To speed up finding a valid swap:
    # For a given c, let Crit_c = { w in N(c) | counts[w] == 1 }
    # We need c' such that N(c') covers Crit_c.
    # This means c' must be within distance 1 of all w in Crit_c.
    
    # We can precompute for each word w, the list of codewords that cover it?
    # No, that's dynamic.
    
    # Let's just try random swaps for codewords with small Crit_c.
    
    # Precompute neighbors of neighbors? No.
    
    # For each c in C:
    #   Crit_c = neighbors[c][counts[neighbors[c]] == 1]
    #   If Crit_c is empty, it's removable (handled above).
    #   If len(Crit_c) is small (e.g., < 3), try to find c'.
    
    # How to find c'?
    # c' must be in intersection of Balls(Crit_c, 1).
    # We can iterate through all 6561 words and check if they are in the intersection.
    # This is 6561 * 17 checks per candidate c.
    # If we do this for all c, it's 400 * 6561 * 17 ~ 45M ops. Might be too slow in Python loop.
    # But we can vectorize?
    
    # Let's try a limited number of swap attempts.
    
    swap_attempts = 0
    max_swap_attempts = 5000
    
    while time.time() < deadline - 2.0 and swap_attempts < max_swap_attempts:
        # Pick a random codeword to try to swap
        # Prefer those with fewer critical neighbors
        # Calculate critical counts for all?
        # crit_counts = (counts[neighbors] == 1).sum(axis=1)
        # This is 6561 * 17 ops. Fast.
        
        crit_counts = (counts[neighbors] == 1).sum(axis=1)
        
        # Only consider codewords in C
        # We want to pick c from C with low crit_counts
        # Get indices of C
        c_indices = np.array(list(selected_set))
        
        if len(c_indices) == 0:
            break
            
        # Filter crit_counts to only selected codewords
        c_crit_counts = crit_counts[c_indices]
        
        # Pick the one with minimum critical count
        # If min is 0, it should have been removed. So min >= 1.
        min_idx_in_c = np.argmin(c_crit_counts)
        c = c_indices[min_idx_in_c]
        
        # If the minimum critical count is too high, swapping is unlikely to succeed easily
        # But let's try.
        
        # Find critical neighbors of c
        c_neighbors = neighbors[c]
        crit_neighbors = c_neighbors[counts[c_neighbors] == 1]
        
        # We need a new codeword c' such that all crit_neighbors are in N(c')
        # This is equivalent to: for all w in crit_neighbors, d(c', w) <= 1
        # Which means c' is in the intersection of N(w) for w in crit_neighbors.
        
        # Let's find the intersection.
        # Start with all words
        # Intersect with N(w) for each w in crit_neighbors
        
        # Optimization: If len(crit_neighbors) is 0, we can swap with anything?
        # No, if len is 0, it's removable.
        
        # We can iterate through all words to find a valid c'.
        # To make it faster, we can restrict to words not in C.
        
        # Let's try to find ANY valid c'.
        # valid_mask = np.ones(total_words, dtype=bool)
        # for w in crit_neighbors:
        #     valid_mask &= np.zeros(total_words, dtype=bool) # wait, need to mark N(w)
            
        # Better:
        # candidates = np.ones(total_words, dtype=bool)
        # for w in crit_neighbors:
        #     # Mark neighbors of w as True, others False? No.
        #     # We want c' such that c' in N(w).
        #     # So candidates &= (neighbors matrix row w is 1?)
        #     # We don't have a full matrix.
        
        # Let's just iterate through all words and check.
        # But 6561 checks * 17 distance calcs is ~100k ops.
        # If we do this 5000 times, it's 500M ops. Too slow in Python.
        
        # We need a faster check.
        # Precompute for each word w, the list of words that are distance 1 away? That's neighbors[w].
        # So c' is valid if for all w in crit_neighbors, c' is in neighbors[w].
        
        # We can use a "voting" scheme or bitsets.
        # Bitsets: 6561 bits = 820 bytes.
        # 6561 bitsets = 5.4 MB.
        # We can represent neighbors as bitsets.
        
        # Let's try to implement bitset intersection.
        
    # Lagrangian Relaxation with Subgradient Optimization
    # Dual variables u[w] penalize under-covered vertices
    # Objective: minimize |C| + sum_w u[w] * max(0, 1 - count[w])
    # Subgradient: for each w, if count[w] == 0, gradient += 1; if count[w] >= 1, gradient -= u[w]
    
    u = np.zeros(total_words, dtype=np.float64)  # Dual variables
    step_size = 0.5
    best_u = u.copy()
    best_count = len(selected_set)
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        # Compute current coverage deficit
        deficit = np.maximum(0, 1 - counts.astype(np.float64))
        
        # Subgradient of the Lagrangian dual function
        # g[w] = 1 if count[w] == 0 else -u[w]
        g = np.where(counts == 0, 1.0, -u)
        
        # Update dual variables
        # Step size: standard subgradient method
        # alpha_k = beta / sqrt(k) or constant
        u = np.maximum(0, u + step_size * g)
        
        # Primal heuristic: score each potential codeword
        # For each word c, compute its "value" = number of deficit vertices it covers
        # minus penalty for already well-covered vertices
        # value[c] = sum_{w in N(c)} u[w] * (1 if count[w] == 0 else 0) - lambda * (number of w in N(c) with count[w] >= 2)
        
        # More simply: marginal benefit of adding c
        # benefit[c] = sum_{w in N(c)} max(0, 1 - count[w])
        # cost[c] = 1 (adding one codeword)
        
        # For swaps: we want to remove c_old and add c_new
        # Net benefit = benefit(c_new) - benefit(c_old) where benefit accounts for what's uniquely covered
        
        # Let's use a simpler scoring:
        # score[c] = sum_{w in N(c)} u[w]
        # This encourages selecting codewords that cover high-dual (under-covered) regions
        
        scores = np.zeros(total_words, dtype=np.float64)
        # Vectorized: for each c, score = sum of u over neighbors[c]
        # This is a matrix-vector product: scores = neighbors_onehot @ u
        # But we don't have a one-hot matrix. We have neighbors as indices.
        # scores[c] = sum(neighbors[c] mapped through u)
        
        # Use advanced indexing
        scores = np.sum(u[neighbors], axis=1)
        
        # Exclude already selected codewords from candidates
        c_indices = np.array(list(selected_set))
        if len(c_indices) == 0:
            break
        
        # For each selected codeword, find the best swap
        # A swap (remove c_old, add c_new) is beneficial if:
        # score[c_new] > score[c_old] AND the swap maintains coverage
        
        # Let's try a different approach:
        # 1. Find the codeword with lowest "importance" (fewest unique contributions)
        # 2. Find the best replacement that covers its critical neighbors
        
        # Importance: number of vertices where this codeword is the ONLY cover
        importance = (counts[neighbors] == 1).sum(axis=1)
        
        # Pick the least important codeword
        c_old_importance = importance[c_indices]
        min_imp_idx = np.argmin(c_old_importance)
        c_old = c_indices[min_imp_idx]
        
        # If importance is 0, it's removable
        if c_old_importance[min_imp_idx] == 0:
            if can_remove(c_old):
                remove_codeword(c_old)
                best_count = len(selected_set)
                no_improve_count = 0
                # Reset u after removal
                u = np.zeros(total_words, dtype=np.float64)
                step_size = 0.5
                continue
        
        # Find best c_new: maximize score[c_new] subject to covering all critical neighbors of c_old
        c_old_neighbors = neighbors[c_old]
        crit_neighbors = c_old_neighbors[counts[c_old_neighbors] == 1]
        
        if len(crit_neighbors) == 0:
            # Can remove
            if can_remove(c_old):
                remove_codeword(c_old)
                best_count = len(selected_set)
                no_improve_count = 0
                u = np.zeros(total_words, dtype=np.float64)
                step_size = 0.5
                continue
        
        # Find candidates that cover all critical neighbors
        # A word c' covers w if c' in N(w), i.e., d(c', w) <= 1
        # We need c' in intersection of N(w) for all w in crit_neighbors
        
        # Build the intersection using set operations
        potential = set(neighbors[crit_neighbors[0]].tolist())
        for w in crit_neighbors[1:]:
            potential &= set(neighbors[w].tolist())
        
        potential.discard(c_old)
        potential -= selected_set
        
        if not potential:
            # No valid swap; decay dual variables and continue
            u *= 0.9
            no_improve_count += 1
            if no_improve_count > 50:
                break
            continue
        
        # Pick the best c' by score
        potential_list = list(potential)
        potential_scores = scores[potential_list]
        best_new_idx = np.argmax(potential_scores)
        c_new = potential_list[best_new_idx]
        
        # Check if this swap is beneficial
        # Compare scores
        if scores[c_new] > scores[c_old]:
            swap_codeword(c_old, c_new)
            best_count = len(selected_set)
            no_improve_count = 0
            # Decay u after a successful swap
            u *= 0.8
            step_size = max(0.1, step_size * 0.95)
        else:
            # Not beneficial; decay and try different approach
            u *= 0.9
            no_improve_count += 1
            if no_improve_count > 100:
                break
    
    # After Lagrangian phase, do a final removal pass
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        current_list = list(selected_set)
        np.random.shuffle(current_list)
        for c in current_list:
            if c not in selected_set: continue
            if can_remove(c):
                remove_codeword(c)
                improved = True
                break
        
    # Final removal pass
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        current_list = list(selected_set)
        np.random.shuffle(current_list)
        for c in current_list:
            if c not in selected_set: continue
            if can_remove(c):
                remove_codeword(c)
                improved = True
                break

    codewords = [idx_to_word(i) for i in selected_set]
    return codewords
