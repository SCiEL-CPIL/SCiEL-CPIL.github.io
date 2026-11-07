def construct():
    import time
    import random
    
    deadline = time.time() + 85
    n = 15
    
    def build_adj_deg(edges):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def check_full(edges):
        adj, deg = build_adj_deg(edges)
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def check_incremental(edges, new_edge):
        """Check if adding new_edge to edges maintains validity.
        Only need to check paths involving the new edge's endpoints."""
        u, v = new_edge
        adj, deg = build_adj_deg(edges)
        # Add the new edge temporarily
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
        # Check all paths s-u-v-t and s-v-u-t
        # For path s-u-v-t: s in adj[u] (s != v), t in adj[v] (t != u, t != s)
        for s in list(adj[u]):
            if s == v:
                continue
            for t in list(adj[v]):
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # For path s-v-u-t: s in adj[v] (s != u), t in adj[u] (t != v, t != s)
        for s in list(adj[v]):
            if s == u:
                continue
            for t in list(adj[u]):
                if t == v or t == s:
                    continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def local_search(edges, max_iters=2000):
        """Try to add more edges via local search."""
        edges = set(edges)
        adj, deg = build_adj_deg(edges)
        
        # Build candidate list: all non-edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        candidates = [(i, j) for i, j in all_pairs if (i, j) not in edge_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for e in candidates:
                if time.time() > deadline - 5:
                    break
                u, v = e
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if check_incremental(edges, e):
                    edges.add(e)
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    adj, deg = build_adj_deg(edges)
                    improved = True
        return edges
    
    def edge_swap(edges, max_iters=500):
        """Try edge swaps: remove one edge, add another, if it improves edge count or enables more additions."""
        edges = set(edges)
        adj, deg = build_adj_deg(edges)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        current_count = len(edges)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to remove
            existing = list(edge_set)
            if not existing:
                break
            u, v = random.choice(existing)
            
            # Pick a random non-edge to add
            non_edges = [(i, j) for i, j in all_pairs if (i, j) not in edge_set]
            if not non_edges:
                break
            a, b = random.choice(non_edges)
            
            # Try the swap
            test_edges = set(edges)
            test_edges.discard((u, v))
            test_edges.discard((v, u))
            test_edges.add((a, b))
            
            if check_full(test_edges):
                # Now try to add more edges
                test_edges = local_search(test_edges, max_iters=100)
                if len(test_edges) > current_count:
                    edges = test_edges
                    current_count = len(edges)
                    adj, deg = build_adj_deg(edges)
                    edge_set = set()
                    for uu, vv in edges:
                        edge_set.add((uu, vv))
                        edge_set.add((vv, uu))
        
        return edges
    
    # Phase 1: Start with K_{7,8} bipartite
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for u in A:
        for v in B:
            base_edges.add((u, v))
    
    # Verify base is valid
    assert check_full(base_edges), "Base K_{7,8} should be valid"
    
    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Phase 2: Multiple random restarts with greedy + local search
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        
        # Start from base
        trial_edges = set(base_edges)
        
        # Greedy addition of intra-part edges
        intra_candidates = []
        for i in range(7):
            for j in range(i+1, 7):
                intra_candidates.append((i, j))
        for i in range(7, 15):
            for j in range(i+1, 15):
                intra_candidates.append((i, j))
        
        random.shuffle(intra_candidates)
        
        for e in intra_candidates:
            if time.time() > deadline - 10:
                break
            if e in trial_edges:
                continue
            if check_incremental(trial_edges, e):
                trial_edges.add(e)
        
        # Local search to add more
        trial_edges = local_search(trial_edges, max_iters=500)
        
        if len(trial_edges) > best_count:
            best_count = len(trial_edges)
            best_edges = set(trial_edges)
    
    # Phase 3: Try edge swaps from best solution
    if time.time() < deadline - 10:
        swapped = edge_swap(best_edges, max_iters=200)
        if len(swapped) > best_count:
            best_count = len(swapped)
            best_edges = set(swapped)
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 3:
        final = local_search(best_edges, max_iters=1000)
        if len(final) > best_count:
            best_count = len(final)
            best_edges = set(final)
    
    # Convert to required format
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
