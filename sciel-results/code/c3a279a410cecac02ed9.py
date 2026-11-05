def construct():
    import math
    import time

    start_time = time.time()
    limit = 10000
    deadline = start_time + 80  # safety margin

    # Generate primes up to 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [primes[0]]

    # states[i] is a list of (length, last_gap) pairs.
    # We maintain only non-dominated states:
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2.
    # We want to keep the Pareto frontier.
    
    # To speed up, we can store states as a list of (L, G).
    # Since L is small (max ~20-30?), and G is integer.
    
    states = [[] for _ in range(n)]
    states[0] = [(1, 0)]
    
    # prev[i] will store the index of the previous prime in the best sequence ending at i.
    # Since we might have multiple states, we need to store the predecessor for the state that leads to the global max.
    # Actually, we can just store the full path or backtrack via a map.
    # Let's store: best_path[i] = (length, gap, prev_index, prev_length, prev_gap) ? No, that's complex.
    # Let's just store the states, and at the end, find the best one and backtrack by re-computing or storing pointers.
    
    # To backtrack, we can store for each state (i, L, G) the predecessor (j, L-1, G_prev).
    # Since states are pruned, we can map (L, G) to prev.
    # Let's use a dictionary for states[i]: key=(L, G), value=prev_index.
    # But multiple (L,G) might come from same prev? No, gap is determined by i and j.
    # So (L, G) uniquely identifies the transition from j?
    # G = p[i] - p[j]. So j is determined by G and i.
    # So we just need to know which j was used for a given (L, G) state at i.
    # But we might have multiple j's leading to same (L, G)?
    # If j1 != j2, then p[j1] != p[j2], so G1 != G2.
    # So G uniquely identifies j (for fixed i).
    # So we can store prev_index for each state.
    
    # Let's refine:
    # states[i] = list of (L, G, prev_idx)
    
    states[0] = [(1, 0, -1)]
    
    for i in range(1, n):
        pi = primes[i]
        current_states = []
        
        # Option 1: Start new sequence at i
        current_states.append((1, 0, -1))
        
        # Option 2: Extend sequences from j < i
        for j in range(i):
            pj = primes[j]
            gap = pi - pj
            # If gap is 0, impossible for distinct primes
            for L, G, _ in states[j]:
                if gap >= G:
                    # Valid extension
                    new_L = L + 1
                    new_G = gap
                    # We need to check if this state is dominated by existing ones in current_states
                    # But we are building the list first, then pruning.
                    current_states.append((new_L, new_G, j))
                    
        # Prune current_states to non-dominated
        # Sort by L descending, then G ascending
        # We want to keep (L, G) if no other (L', G') has L' >= L and G' <= G.
        # Sorting by L desc, G asc:
        # Iterate, keep if G is strictly less than the min G seen so far for higher/equal L.
        
        # Since L is small, we can group by L.
        # Let's just sort and filter.
        if len(current_states) > 1:
            current_states.sort(key=lambda x: (-x[0], x[1]))
            
            filtered = []
            min_gap_for_len_or_more = float('inf')
            # We need to be careful: 
            # If we have (5, 10) and (5, 5), (5,5) dominates (5,10).
            # If we have (5, 10) and (4, 2), neither dominates the other?
            # (5,10) vs (4,2): 5>4 but 10>2. No dominance.
            # So we keep both.
            
            # Algorithm for pruning:
            # Sort by L desc.
            # For a given L, keep only the smallest G.
            # Then, for decreasing L, if G is smaller than the minimum G seen in higher Ls, keep it.
            
            best_gaps = {} # L -> min G
            for L, G, idx in current_states:
                if L not in best_gaps or G < best_gaps[L]:
                    best_gaps[L] = G
            
            # Now filter based on dominance across L
            # We iterate L from max to min.
            # A state (L, G) is useful if G < min(G' for all L' > L).
            # Actually, if there is a state (L', G') with L' >= L and G' <= G, then (L, G) is dominated.
            
            max_L = max(best_gaps.keys())
            min_G_so_far = float('inf')
            final_states = []
            
            for L in range(max_L, 0, -1):
                if L in best_gaps:
                    G = best_gaps[L]
                    if G < min_G_so_far:
                        # Find the index associated with this (L, G)
                        # We need to recover the index.
                        # We lost the index in best_gaps.
                        # Let's re-iterate original list to find the one matching (L, G) that is not dominated?
                        # Actually, if (L, G) is in best_gaps and passes the dominance check, 
                        # there might be multiple j's giving same (L, G)? No, G determines j.
                        # So there is only one j for a given G.
                        # So we can find it.
                        pass # Placeholder
            
            # Simpler pruning:
            # 1. Group by L, keep min G for each L.
            # 2. Sort unique Ls descending.
            # 3. Keep (L, G) if G is strictly less than all Gs of L' > L.
            
            # Let's implement this properly.
            # Step 1: Get min G for each L
            min_G_for_L = {}
            idx_for_state = {} # (L, G) -> idx. If multiple, pick any? No, G determines j, so idx is unique for (L,G) if we assume j is unique for G.
            # Wait, if two different j's give same gap? p[i]-p[j1] = p[i]-p[j2] => j1=j2. So yes, unique.
            
            for L, G, idx in current_states:
                if L not in min_G_for_L or G < min_G_for_L[L]:
                    min_G_for_L[L] = G
                    idx_for_state[(L, G)] = idx # Store the idx for the best G for this L
            
            # Step 2: Filter
            sorted_Ls = sorted(min_G_for_L.keys(), reverse=True)
            min_G_seen = float('inf')
            valid_states = []
            for L in sorted_Ls:
                G = min_G_for_L[L]
                if G < min_G_seen:
                    valid_states.append((L, G, idx_for_state[(L, G)]))
                    min_G_seen = G
                # If G >= min_G_seen, it's dominated by a higher L with smaller/equal G.
                
            states[i] = valid_states
        else:
            states[i] = current_states
            
        # Time check
        if time.time() > deadline:
            break

    # Find the best state overall
    max_len = 0
    best_i = -1
    best_L = 0
    best_G = 0
    best_prev = -1
    
    for i in range(n):
        for L, G, prev_idx in states[i]:
            if L > max_len:
                max_len = L
                best_i = i
                best_L = L
                best_G = G
                best_prev = prev_idx

    # Backtrack
    seq = []
    curr_i = best_i
    curr_prev = best_prev
    # We need to trace back.
    # We have states[curr_i] containing (best_L, best_G, curr_prev).
    # The previous state is at curr_prev, with length best_L - 1 and gap G_prev.
    # We need to find G_prev.
    # G_prev is the last gap of the sequence ending at curr_prev.
    # It is NOT necessarily the gap between curr_prev and its predecessor?
    # Yes, it is.
    # But we need to identify WHICH state at curr_prev was used.
    # The state at curr_prev must have length best_L - 1.
    # And its gap G_prev must satisfy: best_G >= G_prev.
    # Also, best_G = p[best_i] - p[curr_prev].
    
    # We can reconstruct by looking up the state.
    # At curr_prev, we look for a state with L = best_L - 1.
    # There might be multiple. Which one was used?
    # The one that allowed the transition.
    # The transition condition was gap >= G_prev.
    # Since we kept only non-dominated states, and we picked the one that led to the max length,
    # we can just pick the state at curr_prev with L = best_L - 1 and smallest G?
    # Not necessarily. We need the specific one that was part of the optimal path.
    # However, if multiple states at curr_prev have L = best_L - 1, and all satisfy gap >= G,
    # they all lead to the same length at best_i.
    # But their *future* potential might differ.
    # For backtracking, we just need *a* valid path.
    # So we can pick any valid predecessor state.
    
    path = []
    i = best_i
    L = best_L
    while i != -1:
        path.append(primes[i])
        if L == 1:
            break
        prev_L = L - 1
        # Find a state in states[i_prev] with length prev_L
        # We need to know i_prev. It's best_prev from the state we found.
        # But we didn't store the full history.
        # Let's re-trace carefully.
        
        # We have i and the state (L, G, prev_idx) that we identified as best.
        # So prev_idx is the index of the previous prime.
        # We need to find the state at prev_idx that was used.
        # It must have length L-1.
        # And its gap G_prev must be <= G.
        # Let's find it.
        
        # Get the state details for current i
        # We need to access the state we used.
        # In the loop above, we only stored the *best* state for each i? No, we stored the *valid* states.
        # We identified `best_prev` as the `prev_idx` of the best state at `best_i`.
        # So we know `prev_idx`.
        
        # Now we need to find the state at `prev_idx` with length `L-1`.
        # There might be multiple.
        # We pick the one with the smallest gap? Or the one that was actually used?
        # Since all valid states at `prev_idx` with length `L-1` and gap <= G are valid predecessors,
        # and they all result in the same length L at i, the "value" is the same.
        # For the purpose of returning *a* valid sequence, any such predecessor works.
        # However, to ensure we can continue backtracking, we should pick one that itself has a valid predecessor.
        # All states in `states[prev_idx]` are valid by construction.
        # So we can just pick the first one with length L-1.
        
        # To make it deterministic, pick the one with smallest gap (largest index? no, smallest gap).
        
        found_prev_state = None
        for state in states[prev_idx]:
            if state[0] == prev_L:
                found_prev_state = state
                break
        
        if found_prev_state is None:
            # Should not happen
            break
            
        i = prev_idx
        L = prev_L
        
    path.reverse()
    return path
