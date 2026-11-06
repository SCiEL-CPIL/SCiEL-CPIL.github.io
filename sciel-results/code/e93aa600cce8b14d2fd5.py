import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 16
    core_n = 11
    new_vertices = list(range(core_n, n))
    num_new = len(new_vertices)

    # 1. Build Grötzsch Graph (Vertices 0..10)
    # C5: 0-1-2-3-4-0
    # Mycielski:
    # Copies: 5..9 (5 is copy of 0, ..., 9 is copy of 4)
    # Top: 10
    core_edges = []
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        core_edges.append([u, v])
    
    for u, v in c5_edges:
        core_edges.append([u, 5+v])
        core_edges.append([v, 5+u])
        
    for i in range(5):
        core_edges.append([10, 5+i])
        
    # Build adjacency bitmask for core
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)
        
    # 2. Find all independent sets in the core
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind:
            independent_sets.append(mask)
            
    # Precompute popcount for all masks up to 2^11
    popcount = [0] * (1 << core_n)
    for i in range(1, 1 << core_n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Precompute pairs for new vertices
    pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    pair_index = {}
    for k, (i, j) in enumerate(pairs):
        pair_index[(i, j)] = k
        pair_index[(j, i)] = k

    def evaluate_state(sets, new_edges_mask):
        # sets: list of 5 bitmasks
        # new_edges_mask: 10-bit integer
        score = len(core_edges)
        for s in sets:
            score += popcount[s]
        score += popcount[new_edges_mask]
        
        # Check validity
        # 1. Triangle-free in new vertices
        # Build adjacency for new vertices
        adj_new = [0] * num_new
        for k, (i, j) in enumerate(pairs):
            if new_edges_mask & (1 << k):
                adj_new[i] |= (1 << j)
                adj_new[j] |= (1 << i)
                
        for u in range(num_new):
            for v in range(u+1, num_new):
                if adj_new[u] & (1 << v):
                    if adj_new[u] & adj_new[v]:
                        return -1 # Triangle in new vertices
                        
        # 2. Disjointness for edges
        for k, (i, j) in enumerate(pairs):
            if new_edges_mask & (1 << k):
                if sets[i] & sets[j]:
                    return -1 # Triangle via core
                    
        return score

    # Simulated Annealing with multiple restarts
    best_score = -1
    best_state = None
    
    rng = random.Random(42)
    
    def run_sa_iteration(start_sets, start_new_edges, T_start, T_min, alpha, time_limit):
        """Run one SA iteration, return (best_score, best_state)"""
        current_sets = list(start_sets)
        current_new_edges = start_new_edges
        
        # Optimize new_edges for current sets
        local_best_score = -1
        local_best_mask = 0
        for mask in range(1 << 10):
            s = evaluate_state(current_sets, mask)
            if s > local_best_score:
                local_best_score = s
                local_best_mask = mask
                
        current_new_edges = local_best_mask
        iter_best_score = local_best_score
        iter_best_state = (list(current_sets), local_best_mask)
        
        T = T_start
        while T > T_min and time.time() < time_limit:
            # Perturb
            new_sets = list(current_sets)
            
            # Change one set
            i = rng.randint(0, num_new - 1)
            new_sets[i] = rng.choice(independent_sets)
            
            # Re-optimize new_edges for new_sets
            local_best_score = -1
            local_best_mask = 0
            for mask in range(1 << 10):
                s = evaluate_state(new_sets, mask)
                if s > local_best_score:
                    local_best_score = s
                    local_best_mask = mask
                    
            delta = local_best_score - iter_best_score
            
            if delta >= 0 or rng.random() < pow(2.71828, delta / T):
                current_sets = new_sets
                current_new_edges = local_best_mask
                iter_best_score = local_best_score
                
                if local_best_score > iter_best_score:
                    iter_best_score = local_best_score
                    iter_best_state = (list(current_sets), local_best_mask)
                    
            T *= alpha
            
        return iter_best_score, iter_best_state

    # Run multiple SA restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            # First iteration: random initial
            start_sets = [rng.choice(independent_sets) for _ in range(num_new)]
            start_new_edges = 0
        else:
            # Subsequent iterations: start from best found so far with perturbation
            if best_state is not None:
                base_sets, _ = best_state
                start_sets = list(base_sets)
                # Perturb a few sets
                for _ in range(2):
                    i = rng.randint(0, num_new - 1)
                    start_sets[i] = rng.choice(independent_sets)
                start_new_edges = 0
            else:
                start_sets = [rng.choice(independent_sets) for _ in range(num_new)]
                start_new_edges = 0
        
        # Time budget per restart
        remaining = deadline - time.time()
        time_limit = time.time() + remaining / (num_restarts - r)
        
        iter_score, iter_state = run_sa_iteration(start_sets, start_new_edges, 50.0, 1.0, 0.999, time_limit)
        
        if iter_score > best_score:
            best_score = iter_score
            best_state = iter_state

    if best_state is None:
        return core_edges
        
    sets, new_edges_mask = best_state
    
    # Construct final edges
    final_edges = list(core_edges)
    
    for i, v_new in enumerate(new_vertices):
        s = sets[i]
        m = s
        while m:
            lsb = m & -m
            v_core = lsb.bit_length() - 1
            final_edges.append([v_new, v_core])
            m ^= lsb
            
    for k, (i, j) in enumerate(pairs):
        if new_edges_mask & (1 << k):
            u = new_vertices[i]
            v = new_vertices[j]
            final_edges.append([u, v])
            
    # Build adjacency for final graph
    adj = [0] * n
    for u, v in final_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Verify triangle-free
    is_tf = True
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                if adj[u] & adj[v]:
                    is_tf = False
                    break
        if not is_tf:
            break
            
    if not is_tf:
        return core_edges
        
    # 3-colorability check with backtracking
    def is_3_colorable(adj):
        degrees = [bin(adj[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = [False] * 3
            neighbors = adj[v]
            m = neighbors
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used[color[u]] = True
                m ^= lsb
                
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)
        
    if is_3_colorable(adj):
        return core_edges
    
    # Final greedy edge addition pass
    # Try to add any remaining edges while maintaining triangle-free and non-3-colorable
    edge_set = set()
    for u, v in final_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency from edge_set
    adj2 = [0] * n
    for u, v in edge_set:
        adj2[u] |= (1 << v)
        adj2[v] |= (1 << u)
    
    # Try adding each missing edge
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in edge_set:
                continue
            
            # Check if adding edge (u,v) would create a triangle
            common = adj2[u] & adj2[v]
            if common:
                continue  # Would create a triangle
            
            # Try adding the edge
            adj2[u] |= (1 << v)
            adj2[v] |= (1 << u)
            
            # Check if graph is still not 3-colorable
            if not is_3_colorable(adj2):
                edge_set.add((u, v))
                final_edges.append([u, v])
            else:
                # Revert
                adj2[u] ^= (1 << v)
                adj2[v] ^= (1 << u)
    
    # Final verification
    adj_final = [0] * n
    for u, v in final_edges:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)
        
    is_tf = True
    for u in range(n):
        for v in range(u+1, n):
            if adj_final[u] & (1 << v):
                if adj_final[u] & adj_final[v]:
                    is_tf = False
                    break
        if not is_tf:
            break
            
    if not is_tf:
        # Fallback: remove the last added edge
        if len(final_edges) > len(core_edges):
            final_edges.pop()
        return core_edges
        
    if is_3_colorable(adj_final):
        return core_edges

    return final_edges
