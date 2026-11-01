import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    # dot = 4 (same), 2 (1 overlap), 0, -2, -4
    # If dot > 2, i.e., dot == 4 (only self) or ... wait.
    # Distance squared = 8 - 2*dot.
    # We need dist^2 >= 4 => 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2.
    # So conflict if dot > 2.
    # For distinct vectors, max dot is 2 (if they share 2 positions with same sign).
    # Wait, if they share 2 positions with same sign, dot = 2. dist^2 = 4. OK.
    # If they share 2 positions with opposite sign, dot = 0. dist^2 = 8. OK.
    # If they share 3 positions, dot can be 3, 1, -1, -3.
    # If dot = 3, dist^2 = 2 < 4. Conflict.
    # If dot = 1, dist^2 = 6. OK.
    # So conflict if dot > 2.
    # Since entries are -1, 0, 1, dot is integer.
    # So conflict if dot >= 3.
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for fast updates
    # neighbors[i] is a list of indices j such that conflicts[i,j] is True
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_fill(in_set, cc):
        """
        Greedily add vertices with cc[v] == 0 to in_set.
        cc[v] is the number of conflicts v has with the current in_set.
        """
        # We want to add v if cc[v] == 0 and v not in in_set.
        # Adding v increases cc[u] for all u in neighbors[v].
        # We can just iterate through all v not in in_set.
        # To be efficient, we can pick a random order or fixed order.
        # Fixed order is deterministic.
        candidates = np.where(~in_set)[0]
        rng.shuffle(candidates)
        
        for v in candidates:
            if cc[v] == 0 and not in_set[v]:
                in_set[v] = True
                for u in neighbors[v]:
                    cc[u] += 1
        return in_set

    def sa_search(initial_in_set, initial_cc, initial_E, time_limit):
        in_set = initial_in_set.copy()
        cc = initial_cc.copy()
        E = initial_E
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # Precompute random indices for speed?
        # Just use rng.integers
        
        while time.time() < time_limit and T > min_T:
            # Pick a move type
            # 50% Add, 20% Remove, 30% Swap
            move_type = rng.random()
            
            if move_type < 0.5:
                # Add
                # Pick v not in in_set
                # We can pick randomly from all, or from those with low cc
                # For simplicity, pick random v
                v = rng.integers(N4)
                if in_set[v]:
                    continue
                dE = cc[v]
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    in_set[v] = True
                    E += dE
                    for u in neighbors[v]:
                        cc[u] += 1
                    # cc[v] remains the same (count of neighbors in set)
            elif move_type < 0.7:
                # Remove
                # Pick v in in_set
                # We need a list of indices in in_set
                # For efficiency, we could maintain a list, but let's just pick random and check
                # Or pick from a precomputed list of active indices?
                # Since in_set changes, maintaining a list is complex.
                # Let's just pick random v.
                v = rng.integers(N4)
                if not in_set[v]:
                    continue
                dE = -cc[v]
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    in_set[v] = False
                    E += dE
                    for u in neighbors[v]:
                        cc[u] -= 1
                    # cc[v] remains the same
            else:
                # Swap
                # Pick v_out in in_set, v_in not in in_set
                # To avoid infinite loops of picking same, we can try a few times
                # Or just pick two randoms
                v_out = rng.integers(N4)
                v_in = rng.integers(N4)
                if not in_set[v_out] or in_set[v_in]:
                    continue
                if v_out == v_in:
                    continue
                
                # Calculate dE
                # dE = -cc[v_out] + (cc[v_in] - conflicts[v_in, v_out])
                # conflicts is a boolean matrix, so conflicts[v_in, v_out] is 0 or 1
                dE = -cc[v_out] + cc[v_in] - conflicts[v_in, v_out]
                
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    # Perform swap
                    in_set[v_out] = False
                    in_set[v_in] = True
                    E += dE
                    
                    # Update cc
                    # For u in neighbors[v_out]: cc[u] -= 1
                    # For u in neighbors[v_in]: cc[u] += 1
                    # But we must be careful with v_out and v_in themselves.
                    
                    # Update neighbors of v_out
                    for u in neighbors[v_out]:
                        cc[u] -= 1
                    # Update neighbors of v_in
                    for u in neighbors[v_in]:
                        cc[u] += 1
                    
                    # Correct cc[v_out] and cc[v_in]
                    # Old cc[v_out] was count of neighbors in x.
                    # New cc[v_out] (not in set) is count of neighbors in x' = x - v_out + v_in.
                    # Neighbors in x' are (neighbors in x - v_in) + v_in.
                    # So new cc[v_out] = old cc[v_out] - C[v_out, v_in] + C[v_out, v_in] = old cc[v_out].
                    # Wait, the loop above already updated cc[v_out] if v_in is neighbor of v_out?
                    # If v_in is in neighbors[v_out], then in "Update neighbors of v_in", we did cc[v_out] += 1.
                    # And in "Update neighbors of v_out", we did cc[v_in] -= 1 (if v_in is neighbor of v_out).
                    # Let's trace.
                    # If C[v_out, v_in] == 1:
                    #   Loop 1 (nbrs of v_out): u=v_in -> cc[v_in] -= 1.
                    #   Loop 2 (nbrs of v_in): u=v_out -> cc[v_out] += 1.
                    #   Net change for cc[v_out]: +1.
                    #   Net change for cc[v_in]: -1.
                    #   Is this correct?
                    #   New cc[v_out] should be old cc[v_out] + 1?
                    #   Old cc[v_out] = sum_{u in x} C[v_out, u].
                    #   New cc[v_out] = sum_{u in x'} C[v_out, u] = sum_{u in x - v_out + v_in} C[v_out, u].
                    #   Since v_out not in x', sum is over x - v_out + v_in.
                    #   x - v_out + v_in = (x - v_out) U {v_in}.
                    #   Sum = sum_{u in x - v_out} C[v_out, u] + C[v_out, v_in].
                    #   Old cc[v_out] = sum_{u in x - v_out} C[v_out, u] + C[v_out, v_out](0) ... wait.
                    #   Old cc[v_out] was sum over u in x, u != v_out.
                    #   So Old cc[v_out] = sum_{u in x - v_out} C[v_out, u].
                    #   So New cc[v_out] = Old cc[v_out] + C[v_out, v_in].
                    #   If C[v_out, v_in] == 1, New = Old + 1.
                    #   The loops gave +1. Correct.
                    
                    #   New cc[v_in] (now in set) = sum_{u in x' - v_in} C[v_in, u] = sum_{u in x - v_out} C[v_in, u].
                    #   Old cc[v_in] = sum_{u in x} C[v_in, u] = sum_{u in x - v_out} C[v_in, u] + C[v_in, v_out].
                    #   So New cc[v_in] = Old cc[v_in] - C[v_in, v_out].
                    #   If C[v_in, v_out] == 1, New = Old - 1.
                    #   The loops gave -1. Correct.
                    
                    # If C[v_out, v_in] == 0:
                    #   Loops don't touch each other.
                    #   New cc[v_out] = Old cc[v_out] + 0.
                    #   New cc[v_in] = Old cc[v_in] - 0.
                    #   Correct.
                    
                    # So the loops are sufficient. No extra correction needed.
            
            T *= cooling_rate
            
            # Periodic polish?
            # Maybe every 1000 steps?
            # Or just rely on the final polish.
            
        return in_set, cc, E

    # Main Loop
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = np.zeros(N4, dtype=bool)
        cc = np.zeros(N4, dtype=np.int32)
        E = 0
        
        # Greedy min-degree init
        alive = np.ones(N4, dtype=bool)
        degrees = np.array([len(n) for n in neighbors], dtype=np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            E += cc[chosen] # Should be 0 initially? No, cc is updated.
            # Wait, greedy init should build an independent set.
            # So we only pick vertices with cc[v] == 0.
            # In min-degree greedy for independent set, we pick a vertex, add it, remove its neighbors.
            # This is different from the SA cc logic.
            # Let's stick to a simple greedy independent set init.
            
            # Reset in_set and cc for init
            in_set = np.zeros(N4, dtype=bool)
            cc = np.zeros(N4, dtype=np.int32)
            E = 0
            
            # Greedy
            available = np.ones(N4, dtype=bool)
            while np.any(available):
                # Pick vertex with min degree among available
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0: break
                degs = np.array([len(neighbors[i]) for i in avail_indices])
                min_d = np.min(degs)
                cands = avail_indices[degs == min_d]
                chosen = cands[rng.integers(len(cands))]
                
                in_set[chosen] = True
                E += cc[chosen] # cc[chosen] should be 0
                for u in neighbors[chosen]:
                    cc[u] += 1
                    available[u] = False
                available[chosen] = False
                
                # Note: cc for chosen is now the number of neighbors in set.
                # But we don't use cc for chosen anymore in this greedy phase.
            
        # 2. SA
        time_left = deadline - 5 - time.time()
        if time_left > 2:
            in_set, cc, E = sa_search(in_set, cc, E, time.time() + time_left * 0.8)
            
        # 3. Polish
        # Greedy zero-conflict polish
        # We want to maximize the independent set size.
        # Start with the current in_set.
        # Remove conflicts? Or just add non-conflicting vertices?
        # The SA set might have conflicts.
        # We should extract a maximal independent set from the SA set first?
        # Or just run greedy_fill on the whole graph, preferring vertices in the SA set.
        
        # Let's try: 
        # 1. Create a new in_set2
        # 2. Add all v from in_set that don't conflict with in_set2.
        # 3. Add all other v that don't conflict with in_set2.
        
        in_set2 = np.zeros(N4, dtype=bool)
        cc2 = np.zeros(N4, dtype=np.int32)
        
        # Order: SA set first, then others
        sa_indices = np.where(in_set)[0]
        other_indices = np.where(~in_set)[0]
        rng.shuffle(sa_indices)
        rng.shuffle(other_indices)
        
        for v in np.concatenate([sa_indices, other_indices]):
            if cc2[v] == 0 and not in_set2[v]:
                in_set2[v] = True
                for u in neighbors[v]:
                    cc2[u] += 1
        
        current_size = in_set2.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set2.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
