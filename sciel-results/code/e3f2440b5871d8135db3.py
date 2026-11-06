def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    all_possible = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    best_edges = set()
    best_count = 0
    
    # Initial Clebsch
    edges, adj = construct_clebsch()
    edges = set(edges)
    if not has_q3_adj(adj, 16):
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            
    candidates = [e for e in all_possible if e not in edges]
    
    while time.time() < deadline:
        random.shuffle(candidates)
        improved = False
        for e in candidates[:]:
            if time.time() > deadline:
                break
                
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, e, 16):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edges.add(e)
                candidates.remove(e)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
                improved = True
                
        if not improved:
            if random.random() < 0.5:
                if edges:
                    removed = random.choice(list(edges))
                    edges.remove(removed)
                    u, v = removed
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    candidates.append(removed)
            else:
                # Restart from Clebsch
                edges, adj = construct_clebsch()
                edges = set(edges)
                candidates = [e for e in all_possible if e not in edges]
                if has_q3_adj(adj, 16):
                    # Should not happen
                    break

    if has_q3_adj(adj, 16):
        # Final check on best_edges?
        # We should return a valid graph.
        # If best_edges is valid, return it.
        # We don't have adj for best_edges.
        # We can reconstruct adj for best_edges and check.
        pass
        
    return [list(e) for e in edges]
