import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
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
    
    # 3. Find Max Independent Set
    best_size = 0
    best_set = set()
    
    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            # Find node with minimum degree within remaining
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            # Remove min_node and its neighbors
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s

    def local_search(init_set, max_iter=200):
        """Local search: try to add nodes, and do 1-swap if stuck."""
        current_set = set(init_set)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute blocked set
            blocked = set()
            for v in current_set:
                blocked.update(adj_list[v])
            
            # Try to add a random unblocked node
            candidates = [v for v in range(N_S) if v not in current_set and v not in blocked]
            
            if candidates:
                v_add = candidates[np.random.randint(len(candidates))]
                current_set.add(v_add)
                continue
            
            # Stuck: try a 1-swap to escape local optimum
            if current_set:
                v_remove = list(current_set)[np.random.randint(len(current_set))]
                # Temporarily remove
                current_set.remove(v_remove)
                
                # Recompute blocked
                blocked2 = set()
                for v in current_set:
                    blocked2.update(adj_list[v])
                
                candidates2 = [v for v in range(N_S) if v not in current_set and v not in blocked2]
                
                if candidates2:
                    # Accept the swap (size stays same, but structure changes)
                    v_add = candidates2[np.random.randint(len(candidates2))]
                    current_set.add(v_add)
                else:
                    # Revert
                    current_set.add(v_remove)
        
        return current_set

    # Run multiple restarts with different random seeds
    np.random.seed(42)
    
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        init_set = greedy_min_degree()
        final_set = local_search(init_set, max_iter=100)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
    
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
