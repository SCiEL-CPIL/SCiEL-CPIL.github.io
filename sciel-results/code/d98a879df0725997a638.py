import numpy as np
import itertools
import time

def construct():
        start_time = time.time()
        deadline = start_time + 145.0
        D = 11

        # 1. Generate all supports (combinations of 4 indices)
        supports = list(itertools.combinations(range(D), 4))
        N_S = len(supports)  # 330

        # Convert supports to bitmasks for fast intersection check
        masks = []
        for s in supports:
            m = 0
            for idx in s:
                m |= (1 << idx)
            masks.append(m)

        # 2. Build conflict graph
        # Conflict if intersection size is 3
        adj = [set() for _ in range(N_S)]
        
        for i in range(N_S):
            for j in range(i + 1, N_S):
                inter = (masks[i] & masks[j]).bit_count()
                if inter == 3:
                    adj[i].add(j)
                    adj[j].add(i)

        # Convert to list and boolean matrix for vectorized ops
        adj_list = [list(s) for s in adj]
        adj_bool = np.zeros((N_S, N_S), dtype=bool)
        for i in range(N_S):
            for j in adj_list[i]:
                adj_bool[i][j] = True
                adj_bool[j][i] = True # Symmetric, though loop above handles i<j, so we set both or rely on symmetry in usage.
                # Actually, the loop above adds i to adj[j] and j to adj[i].
                # So adj_bool should be symmetric.
        
        # Ensure symmetry in adj_bool just in case
        adj_bool = adj_bool | adj_bool.T

        best_size = 0
        best_set_mask = np.zeros(N_S, dtype=bool)
        rng = np.random.RandomState(42)

        def get_size(mask):
            return np.sum(mask)

        def greedy_random():
            """Randomized greedy independent set."""
            order = rng.permutation(N_S)
            selected = np.zeros(N_S, dtype=bool)
            for v in order:
                if not selected[v]:
                    # Check if any neighbor is selected
                    # adj_bool[v] is the row for v.
                    # We need to check if any selected node is in adj_bool[v]
                    # i.e., np.any(selected & adj_bool[v])
                    if not np.any(selected & adj_bool[v]):
                        selected[v] = True
            return selected

        def local_search(init_mask, max_iter=200):
            """
            Local search on boolean mask.
            Tries to add nodes. If stuck, performs 1-swap or random restart of local part.
            """
            current = init_mask.copy()
            best_local = current.copy()
            best_local_size = get_size(best_local)
            
            for _ in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute blocked nodes (neighbors of current set)
                # blocked = OR of rows of adj_bool for nodes in current
                if np.any(current):
                    blocked = np.any(adj_bool[current], axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                
                # Candidates: not in current, not blocked
                candidates = ~(current | blocked)
                cand_indices = np.where(candidates)[0]
                
                if len(cand_indices) > 0:
                    # Greedily add one (or all? Adding one at a time allows more flexibility in ordering, 
                    # but adding all compatible at once is faster. 
                    # However, adding all might block better future adds? 
                    # In MIS, if A and B are compatible, adding both is always better than adding one.
                    # So we can add ALL candidates at once?
                    # Yes, if they are all mutually compatible? 
                    # Wait. Candidates are defined as "not conflicting with CURRENT".
                    # Two candidates might conflict with EACH OTHER.
                    # So we cannot just add all.
                    # We must pick one, update, recompute.
                    # Or, we can pick a random one to keep it simple and stochastic.
                    v_add = int(cand_indices[rng.randint(len(cand_indices))])
                    current[v_add] = True
                    
                    if get_size(current) > best_local_size:
                        best_local_size = get_size(current)
                        best_local = current.copy()
                    continue
                
                # Stuck: No candidates. Try 1-swap.
                # Pick a random node in current to remove.
                current_indices = np.where(current)[0]
                if len(current_indices) == 0:
                    break
                    
                v_remove = int(current_indices[rng.randint(len(current_indices))])
                current[v_remove] = False
                
                # Recompute blocked
                if np.any(current):
                    blocked2 = np.any(adj_bool[current], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2 = ~(current | blocked2)
                cand_indices2 = np.where(candidates2)[0]
                
                if len(cand_indices2) > 0:
                    # Try to add a candidate. 
                    # It might be the same as v_remove or a different one.
                    # If we add a different one, size stays same or increases?
                    # We removed 1. If we add 1, size is same.
                    # We want to increase size. 
                    # So we should try to add MULTIPLE candidates?
                    # Or, if we removed 1, and can add 2, size increases.
                    # Let's try to add as many as possible greedily after the removal.
                    # But that might be slow.
                    # For now, just add one random candidate.
                    v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                    current[v_add] = True
                    
                    # Check if we improved
                    if get_size(current) > best_local_size:
                        best_local_size = get_size(current)
                        best_local = current.copy()
                else:
                    # Couldn't add anything back. Restore v_remove.
                    current[v_remove] = True
                    # If we can't even restore to a state where we can add, we are stuck in a local minimum.
                    # Break and restart?
                    # Or try a different removal.
                    # To avoid infinite loop of failing swaps, we break and let the outer restart handle it.
                    break

            return best_local

        # Main loop
        num_restarts = 2000
        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # 1. Greedy Init
            init_mask = greedy_random()
            
            # 2. Local Search
            final_mask = local_search(init_mask, max_iter=100)
            
            size = get_size(final_mask)
            if size > best_size:
                best_size = size
                best_set_mask = final_mask.copy()
                
            # Optional: If we hit a very good size, we can stop or reduce iterations.
            # But let's keep searching.

        # 3. Construct vectors
        # For each support in best_set_mask, add all 16 sign vectors
        vectors = []
        for s_idx in range(N_S):
            if best_set_mask[s_idx]:
                support = supports[s_idx]
                # Generate all 16 sign combinations
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int8)
                    for idx, sign in zip(support, signs):
                        v[idx] = sign
                    vectors.append(v)
        
        if not vectors:
            return np.array([], dtype=np.int8).reshape(0, D)
            
        final_vectors = np.array(vectors, dtype=np.int8)
        
        return final_vectors
