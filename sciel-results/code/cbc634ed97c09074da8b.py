import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    n = 43
    
    # Fallback graph: 10 disjoint triangles
    # n=30, alpha=10 < 11
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    def check_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u+1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1): # more than 1 bit
                    return False
        return True

    def compute_alpha(adj_mask, n):
        # Max independent set using branch and bound
        best = [0]
        timeout_flag = [False]
        
        def dfs(candidates, size):
            if timeout_flag[0]:
                return
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            # Upper bound
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Time check
            if time.time() > deadline - 2:
                timeout_flag[0] = True
                return
            
            # Heuristic: pick vertex with max degree in candidates?
            # Or min degree?
            # Standard: pick vertex with max degree to reduce candidates fast.
            # But for independent set, picking a vertex with high degree removes many candidates.
            # Let's pick vertex with max degree in the subgraph induced by candidates.
            
            # Find vertex with max degree
            max_deg = -1
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include chosen
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            if timeout_flag[0]:
                return
            
            # Branch 2: Exclude chosen
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    # Try to build a C4-free graph on n=43
    for seed in range(100):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        
        # Generate all possible edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                edges.append((u, v))
        
        random.shuffle(edges)
        
        adj_mask = [0] * n
        graph_edges = []
        
        for u, v in edges:
            # Check if adding (u,v) creates a C4
            # Condition: No x in N(v) and y in N(u) such that x ~ y
            # N(u) = adj_mask[u], N(v) = adj_mask[v]
            # We need to check if there is an edge between N(u) and N(v)
            
            # Iterate over neighbors of v
            # If N(v) is small, iterate over N(v)
            # If N(u) is small, iterate over N(u)
            
            nu = adj_mask[u]
            nv = adj_mask[v]
            
            # Iterate over the smaller neighborhood for efficiency
            safe = True
            if nv.bit_count() <= nu.bit_count():
                c = nv
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
            else:
                c = nu
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            
            if safe:
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                graph_edges.append([u, v])
        
        # Verify C4-free
        if not check_c4_free(adj_mask, n):
            continue
            
        # Compute alpha
        alpha = compute_alpha(adj_mask, n)
        
        if alpha < 11:
            return {'n': n, 'edges': graph_edges}
        
        # If alpha >= 11, try local search: remove edges to reduce alpha
        # Removing edges can't create C4, so C4-free is preserved
        if alpha <= 13 and time.time() < deadline - 30:
            # Try removing random edges to reduce independence number
            for removal_attempt in range(50):
                if time.time() > deadline - 25:
                    break
                
                # Find an independent set of size alpha
                # Use a greedy approach to find a large independent set
                def find_independent_set(adj_mask, n, target_size):
                    """Find an independent set of at least target_size vertices"""
                    remaining = list(range(n))
                    independent = []
                    random.shuffle(remaining)
                    
                    for v in remaining:
                        # Check if v is not adjacent to any in independent
                        can_add = True
                        for u in independent:
                            if (adj_mask[v] >> u) & 1:
                                can_add = False
                                break
                        if can_add:
                            independent.append(v)
                            if len(independent) >= target_size:
                                break
                    
                    return independent if len(independent) >= target_size else None
                
                # Find an independent set
                ind_set = find_independent_set(adj_mask, n, alpha)
                if ind_set is None:
                    break
                
                # Try removing a random edge within the independent set
                if len(ind_set) >= 2:
                    idx1, idx2 = random.sample(range(len(ind_set)), 2)
                    u, v = ind_set[idx1], ind_set[idx2]
                    
                    # Check if (u,v) is currently an edge
                    if (adj_mask[u] >> v) & 1:
                        # Remove the edge
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        graph_edges = [e for e in graph_edges if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
                        
                        # Re-check alpha
                        new_alpha = compute_alpha(adj_mask, n)
                        if new_alpha < 11:
                            return {'n': n, 'edges': graph_edges}
                        alpha = new_alpha
                        if alpha < 11:
                            return {'n': n, 'edges': graph_edges}
            
    # If no graph found on n=43, try n=42, 41, etc?
    # Or just return fallback
    # Given the high probability of success for n=43, fallback is unlikely needed.
    # But to be safe, we can try smaller n if time permits.
    
    for target_n in range(42, 29, -1):
        if time.time() > deadline - 5:
            break
        
        for seed in range(10):
            if time.time() > deadline - 2:
                break
            random.seed(seed + 1000)
            edges = []
            for u in range(target_n):
                for v in range(u+1, target_n):
                    edges.append((u, v))
            random.shuffle(edges)
            
            adj_mask = [0] * target_n
            graph_edges = []
            
            for u, v in edges:
                nu = adj_mask[u]
                nv = adj_mask[v]
                safe = True
                if nv.bit_count() <= nu.bit_count():
                    c = nv
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nu:
                            safe = False
                            break
                else:
                    c = nu
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nv:
                            safe = False
                            break
                if safe:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    graph_edges.append([u, v])
                    
            if check_c4_free(adj_mask, target_n):
                alpha = compute_alpha(adj_mask, target_n)
                if alpha < 11:
                    return {'n': target_n, 'edges': graph_edges}
                    
    return {'n': 30, 'edges': fallback_edges}
