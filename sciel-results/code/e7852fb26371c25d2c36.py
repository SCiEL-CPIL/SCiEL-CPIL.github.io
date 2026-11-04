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
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is >= 3
    # This ensures that for ANY sign assignment, the distance condition holds.
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter >= 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert adj to list of lists for faster access in loops
    adj_list = [list(s) for s in adj]
    
    # Precompute adjacency as boolean matrix for fast vectorized operations
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj_list[i]:
            adj_bool[i][j] = True
            adj_bool[j][i] = True
            
    best_size = 0
    best_set = set()
    
    rng = np.random.RandomState(42)

    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
            
        return s

    def local_search_fast(init_set, max_iter=1000):
        """
        Fast local search using vectorized operations to determine candidates.
        State: current_set (set of indices)
        Move: Add a vertex if possible, or swap (remove one, add another).
        """
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute in_set and blocked using vectorized ops
            in_set = np.zeros(N_S, dtype=bool)
            for v in current_set:
                in_set[v] = True
            
            # blocked = OR of all rows of adj_bool for vertices in current_set
            if current_set:
                blocked = np.any(adj_bool[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Candidates: not in set and not blocked
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) > 0:
                # Greedily add a random candidate
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                continue
            
            # Stuck: try 1-swap (remove one, add another)
            if current_set:
                # Randomly pick a vertex to remove
                v_remove = list(current_set)[rng.randint(len(current_set))]
                current_set.remove(v_remove)
                
                # Recompute blocked without v_remove
                in_set2 = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    in_set2[v] = True
                
                if current_set:
                    blocked2 = np.any(adj_bool[in_set2], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                # Candidates now include v_remove (since it's not in set and not blocked by itself)
                # But v_remove is blocked by its neighbors in the new set?
                # No, blocked2 is computed from current_set (which doesn't have v_remove).
                # So v_remove is a candidate if it's not in current_set and not blocked by current_set.
                # Since v_remove was in the previous set, it might be blocked by other vertices in current_set?
                # If u in current_set is adjacent to v_remove, then blocked2[v_remove] is True.
                # So we just check the mask.
                
                candidates2_mask = ~(in_set2 | blocked2)
                cand_indices2 = np.where(candidates2_mask)[0]
                
                if len(cand_indices2) > 0:
                    # Try to add a candidate. 
                    # We want to add a vertex that is NOT v_remove, to actually improve size?
                    # If we add v_remove, we are back to the same size.
                    # We want to add a different vertex to potentially increase size or maintain size for future moves.
                    # But if we add v_remove, size stays same. If we add other, size increases.
                    # Let's prefer adding a vertex that is not v_remove if possible.
                    # However, if cand_indices2 only has v_remove, we just revert.
                    
                    # Filter out v_remove to see if we can grow
                    other_cands = [c for c in cand_indices2 if c != v_remove]
                    if other_cands:
                        v_add = int(other_cands[rng.randint(len(other_cands))])
                    else:
                        v_add = v_remove # Revert
                    
                    current_set.add(v_add)
                    
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = set(current_set)
                else:
                    # No candidates, revert
                    current_set.add(v_remove)
        
        return best_local

    # Run multiple restarts
    # We can also try different random initializations
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Alternate between greedy min-degree and random greedy
        if r % 2 == 0:
            init_set = greedy_min_degree()
        else:
            # Random greedy
            order = rng.permutation(N_S)
            init_set = set()
            in_set = np.zeros(N_S, dtype=bool)
            for v in order:
                if not np.any(in_set & adj_bool[v]):
                    init_set.add(v)
                    in_set[v] = True
        
        final_set = local_search_fast(init_set, max_iter=200)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
            
            # Early exit if we hit a very high bound? 
            # Max possible IS size is roughly N_S / (avg_degree+1) ?
            # Avg degree in this graph:
            # Each support (4-set) conflicts with supports sharing 3 or 4 indices.
            # Sharing 4: 0 (unique).
            # Sharing 3: Choose 3 from 4 (4 ways), choose 1 from remaining 7 (7 ways). 4*7=28.
            # So degree is 28.
            # N_S = 330. 330 / 29 ~ 11.
            # So max size is likely around 11-12.
            if best_size >= 12:
                # Can't do much better than 12-13 probably.
                # But let's keep searching until time runs out or we find 13.
                pass

    # Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    return final_vectors
