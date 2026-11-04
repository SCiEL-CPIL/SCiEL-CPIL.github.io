"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_supports():
    """Generate all 4-element subsets of range(D) as bitmasks."""
    supports = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    return supports

def _build_adjacency(supports):
    """Build adjacency list for conflict graph (intersection size >= 3)."""
    n = len(supports)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = supports[i] & supports[j]
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _greedy_mis(adj, random_seed=0):
    """Find a large independent set using greedy heuristic with randomization."""
    rng = np.random.default_rng(random_seed)
    n = len(adj)
    in_set = [False] * n
    degree = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    # Copy degree to track remaining degrees? Or just use original?
    # Greedy: pick node with min current degree in remaining graph.
    # Simple version: pick node with min original degree, remove neighbors.
    # Better: dynamic degree.
    
    remaining = set(range(n))
    selected = []
    
    while remaining:
        # Pick node with min degree in remaining
        # For speed, just pick min degree from original if graph is static?
        # No, degree changes.
        # For n=330, O(n^2) is fine.
        min_deg = float('inf')
        best_node = -1
        for u in remaining:
            deg = sum(1 for v in remaining if v in adj[u])
            if deg < min_deg:
                min_deg = deg
                best_node = u
        
        selected.append(best_node)
        remaining.remove(best_node)
        # Remove neighbors
        for v in adj[best_node]:
            if v in remaining:
                remaining.remove(v)
                
    return selected

def _local_search_mis(adj, init_set, max_iter=1000, time_limit=1.0):
    """Improve an independent set using local search."""
    start_time = time.time()
    n = len(adj)
    in_set = set(init_set)
    in_set_bool = np.zeros(n, dtype=bool)
    for i in in_set:
        in_set_bool[i] = True
        
    # conflict_count[i] = number of neighbors of i in in_set
    conflict_count = np.zeros(n, dtype=np.int32)
    for u in in_set:
        for v in adj[u]:
            conflict_count[v] += 1
            
    best_size = len(in_set)
    best_set = set(in_set)
    
    # Precompute neighbors as lists for speed
    adj_lists = [list(s) for s in adj]
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # 1. Try to add any free node
        free_nodes = [i for i in range(n) if not in_set_bool[i] and conflict_count[i] == 0]
        if free_nodes:
            # Pick one, e.g., random or first
            v = free_nodes[np.random.randint(len(free_nodes))]
            in_set.add(v)
            in_set_bool[v] = True
            for u in adj_lists[v]:
                conflict_count[u] += 1
            improved = True
            if len(in_set) > best_size:
                best_size = len(in_set)
                best_set = set(in_set)
            continue # Restart loop to check for more free nodes
        
        # 2. Try swaps
        # For each u in in_set, check if removing u allows adding more nodes
        # Nodes that become free are those with conflict_count == 1 and neighbor is u
        # Let's iterate over u in in_set
        # To avoid O(n^2) per step, we can be smarter, but n=330 is small.
        
        best_swap_gain = 0
        best_swap_u = -1
        best_swap_adds = []
        
        for u in list(in_set):
            # Identify nodes that would become free if u is removed
            # These are v not in in_set such that conflict_count[v] == 1 and u in adj[v]
            candidates = []
            for v in adj_lists[u]:
                if not in_set_bool[v] and conflict_count[v] == 1:
                    candidates.append(v)
            
            if not candidates:
                continue
                
            # We can add a subset of candidates that is independent.
            # Greedily pick from candidates
            # Build subgraph of candidates
            # Since candidates are all neighbors of u, they might be adjacent to each other.
            # We need Max Independent Set of candidates.
            # Candidates size is small (degree of u is ~28).
            # So greedy MIS on candidates is fast.
            
            # Sort candidates by degree in candidate subgraph?
            # Or just random order.
            np.random.shuffle(candidates)
            
            local_set = set()
            local_bool = {v: False for v in candidates}
            
            for v in candidates:
                # Check if v is compatible with already picked in local_set
                compatible = True
                for w in local_set:
                    if w in adj[v]:
                        compatible = False
                        break
                if compatible:
                    local_set.add(v)
                    local_bool[v] = True
            
            gain = len(local_set) - 1
            if gain > best_swap_gain:
                best_swap_gain = gain
                best_swap_u = u
                best_swap_adds = list(local_set)
        
        if best_swap_gain > 0:
            u = best_swap_u
            # Remove u
            in_set.remove(u)
            in_set_bool[u] = False
            for v in adj_lists[u]:
                conflict_count[v] -= 1
            
            # Add best_swap_adds
            for v in best_swap_adds:
                in_set.add(v)
                in_set_bool[v] = True
                for w in adj_lists[v]:
                    conflict_count[w] += 1
            
            improved = True
            if len(in_set) > best_size:
                best_size = len(in_set)
                best_set = set(in_set)
            
        if not improved:
            # Stuck. Try random restart or break.
            # For now, break to allow new restart.
            break
            
    return best_set

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports
    supports = _generate_supports()
    n_sup = len(supports)
    
    # 2. Build adjacency
    adj = _build_adjacency(supports)
    
    # 3. Find large MIS
    best_mis = set()
    best_size = 0
    
    # Multiple restarts
    num_restarts = 50
    time_per_restart = (deadline - time.time()) / num_restarts
    if time_per_restart < 0.1:
        time_per_restart = 0.1
        num_restarts = max(1, int((deadline - time.time()) / time_per_restart))
        
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Greedy init
        init = _greedy_mis(adj, random_seed=i)
        
        # Local search
        time_limit_for_ls = min(time_per_restart, 2.0)
        mis = _local_search_mis(adj, init, max_iter=500, time_limit=time_limit_for_ls)
        
        if len(mis) > best_size:
            best_size = len(mis)
            best_mis = mis
            
    # 4. Generate vectors from best MIS
    vectors = []
    for idx in best_mis:
        mask = supports[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
    
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Add Shell 2 vectors
    # Generate all shell 2 vectors
    shell2_vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations
            shell2_vecs.append(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][:D] * 0)) # Placeholder
            # Better:
            v1 = np.zeros(D, dtype=np.int64); v1[i]=1; v1[j]=1
            v2 = np.zeros(D, dtype=np.int64); v2[i]=1; v2[j]=-1
            v3 = np.zeros(D, dtype=np.int64); v3[i]=-1; v3[j]=1
            v4 = np.zeros(D, dtype=np.int64); v4[i]=-1; v4[j]=-1
            shell2_vecs.extend([v1, v2, v3, v4])
            
    shell2 = np.array(shell2_vecs, dtype=np.int64)
    
    if len(S4) > 0:
        # Check compatibility
        # u in S4, v in S2. Need u.v <= 1.
        dots = shell2 @ S4.T
        max_dots = dots.max(axis=1)
        valid_2_mask = max_dots <= 1
        S2_candidates = shell2[valid_2_mask]
        
        if len(S2_candidates) > 0:
            # All distinct S2 vectors are mutually compatible (dot <= 1 < 2)
            # So we can take all of them.
            result = np.vstack([S4, S2_candidates])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
