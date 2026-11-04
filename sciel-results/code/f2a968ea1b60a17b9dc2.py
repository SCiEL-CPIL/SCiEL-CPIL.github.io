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
    # Conflict if intersection size is 3
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert adj to list of lists for faster access
    adj_list = [list(s) for s in adj]
    
    # Precompute adjacency as boolean matrix for fast vectorized operations
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj_list[i]:
            adj_bool[i, j] = True
    # Ensure symmetry
    adj_bool = adj_bool | adj_bool.T

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

    def greedy_randomized(rng):
        """Randomized greedy: pick among nodes with minimum degree."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_nodes = []
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            min_node = min_nodes[rng.randint(len(min_nodes))]
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s

    def local_search_fast(init_set, max_iter=300, rng=None):
        """Fast local search with vectorized blocked set computation.
        Tracks best solution found during search."""
        if rng is None:
            rng = np.random
        
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 8.0:
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
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                continue
            
            # Stuck: try a 1-swap to escape local optimum
            if current_set:
                v_remove = list(current_set)[rng.randint(len(current_set))]
                current_set.remove(v_remove)
                
                # Recompute blocked
                in_set2 = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    in_set2[v] = True
                
                if current_set:
                    blocked2 = np.any(adj_bool[in_set2], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2_mask = ~(in_set2 | blocked2)
                cand_indices2 = np.where(candidates2_mask)[0]
                
                if len(cand_indices2) > 0:
                    v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                    current_set.add(v_add)
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = set(current_set)
                else:
                    # Revert
                    current_set.add(v_remove)
                continue
            
            # Empty set: just add a random node
            v_add = int(rng.randint(N_S))
            current_set.add(v_add)
            if len(current_set) > best_local_size:
                best_local_size = len(current_set)
                best_local = set(current_set)
        
        return best_local

    # Run multiple restarts with different random seeds
    best_size = 0
    best_set = set()
    
    # Phase 1: Many quick restarts
    rng = np.random.RandomState(42)
    
    # First, try deterministic greedy
    init_set = greedy_min_degree()
    final_set = local_search_fast(init_set, max_iter=200, rng=rng)
    size = len(final_set)
    if size > best_size:
        best_size = size
        best_set = set(final_set)
    
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Alternate between deterministic and randomized greedy
        if r % 3 == 0:
            init_set = greedy_min_degree()
        else:
            init_set = greedy_randomized(rng)
        
        final_set = local_search_fast(init_set, max_iter=100, rng=rng)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
    
    # Phase 2: Polish the best solution with more iterations
    if time.time() < deadline - 10.0:
        polished = local_search_fast(best_set, max_iter=500, rng=rng)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = set(polished)
    
    # Phase 3: Final polish if time remains
    if time.time() < deadline - 5.0:
        polished2 = local_search_fast(best_set, max_iter=300, rng=rng)
        if len(polished2) > best_size:
            best_size = len(polished2)
            best_set = set(polished2)
    
    # 4. Construct vectors
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
