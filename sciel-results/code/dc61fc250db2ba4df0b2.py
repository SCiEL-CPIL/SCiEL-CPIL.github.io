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
            adj_bool[j][i] = True
    
    # Ensure symmetry in adj_bool just in case
    adj_bool = adj_bool | adj_bool.T

    best_size = 0
    best_set_mask = np.zeros(N_S, dtype=bool)
    rng = np.random.RandomState(42)

    def get_size(mask):
        return int(np.sum(mask))

    def greedy_random():
        """Randomized greedy independent set."""
        order = rng.permutation(N_S)
        selected = np.zeros(N_S, dtype=bool)
        for v in order:
            if not selected[v]:
                if not np.any(selected & adj_bool[v]):
                    selected[v] = True
        return selected

    def local_search(init_mask, max_iter=500):
        """
        Local search on boolean mask.
        Tries to add nodes. If stuck, performs 1-swap or 2-swap.
        """
        current = init_mask.copy()
        best_local = current.copy()
        best_local_size = get_size(best_local)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute blocked nodes (neighbors of current set)
            if np.any(current):
                blocked = np.any(adj_bool[current], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Candidates: not in current, not blocked
            candidates = ~(current | blocked)
            cand_indices = np.where(candidates)[0]
            
            if len(cand_indices) > 0:
                # Greedily add one
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current[v_add] = True
                
                if get_size(current) > best_local_size:
                    best_local_size = get_size(current)
                    best_local = current.copy()
                continue
            
            # Stuck: No candidates. Try 1-swap.
            current_indices = np.where(current)[0]
            if len(current_indices) == 0:
                break
                
            improved = False
            
            # Try 1-swap
            for _ in range(10): # Try a few 1-swaps
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
                    # Try to add a candidate
                    v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                    current[v_add] = True
                    
                    new_size = get_size(current)
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = current.copy()
                        improved = True
                        current_indices = np.where(current)[0]
                        break
                    else:
                        # Revert swap if no improvement
                        current[v_add] = False
                        current[v_remove] = True
                        current_indices = np.where(current)[0]
            
            if improved:
                continue
            
            # Try 2-swap
            if len(current_indices) >= 2:
                # Pick two distinct nodes to remove
                i1, i2 = rng.choice(len(current_indices), replace=False, size=2)
                v_rem1 = int(current_indices[i1])
                v_rem2 = int(current_indices[i2])
                
                current[v_rem1] = False
                current[v_rem2] = False
                
                # Recompute blocked
                if np.any(current):
                    blocked3 = np.any(adj_bool[current], axis=0)
                else:
                    blocked3 = np.zeros(N_S, dtype=bool)
                
                candidates3 = ~(current | blocked3)
                cand_indices3 = np.where(candidates3)[0]
                
                # We want to add at least 2 nodes to improve size
                # Or even 1 node if we removed 2? No, that decreases size.
                # We need to add >= 2 nodes.
                # Let's greedily add 2 compatible nodes from candidates3.
                
                added = []
                # Filter candidates that are compatible with each other and current
                # We can just iterate and pick greedily
                temp_current = current.copy()
                
                # Try to add 2
                for _ in range(2):
                    if len(cand_indices3) == 0:
                        break
                    # Update candidates based on temp_current
                    # Actually, let's just pick from cand_indices3 and check compatibility with temp_current
                    # And compatibility with previously added in this step
                    
                    valid_cands = []
                    for c in cand_indices3:
                        # Check if c is compatible with temp_current
                        if not np.any(temp_current & adj_bool[c]):
                            valid_cands.append(c)
                    
                    if not valid_cands:
                        break
                        
                    v_add = int(valid_cands[rng.randint(len(valid_cands))])
                    temp_current[v_add] = True
                    added.append(v_add)
                    
                    # Remove v_add from cand_indices3? 
                    # Actually, we just need to ensure the next one is compatible with v_add.
                    # The check `np.any(temp_current & adj_bool[c])` handles that.
                
                if len(added) >= 2:
                    # Apply the changes
                    for v in added:
                        current[v] = True
                    
                    new_size = get_size(current)
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = current.copy()
                        improved = True
                        current_indices = np.where(current)[0]
                
                if not improved:
                    # Revert 2-swap
                    for v in added:
                        current[v] = False
                    current[v_rem1] = True
                    current[v_rem2] = True
                    current_indices = np.where(current)[0]
            
            if not improved:
                # If we couldn't improve with 1-swap or 2-swap, break
                # We could try more random restarts of local search, but let's just break
                # and let the outer loop handle it
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
        final_mask = local_search(init_mask, max_iter=200)
        
        size = get_size(final_mask)
        if size > best_size:
            best_size = size
            best_set_mask = final_mask.copy()

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
