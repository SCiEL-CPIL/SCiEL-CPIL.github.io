def ac3_optimized(domains, adj):
    queue = []
    for u in range(n):
        for v in adj[u]:
            queue.append((u, v))
            
    while queue:
        u, v = queue.pop(0)
        
        # Optimization: If domains[u] has more than 1 value, 
        # it cannot prune domains[v] because for any val in domains[v], 
        # there is some other value in domains[u] that is != val.
        if len(domains[u]) > 1:
            continue
            
        # If len(domains[u]) == 0, domains[v] should have been pruned earlier? 
        # If domains[u] is empty, the instance is unsatisfiable.
        if not domains[u]:
            return False
            
        # len(domains[u]) == 1
        u_val = next(iter(domains[u]))
        
        # If u_val is in domains[v], it has no support (since u only has u_val)
        if u_val in domains[v]:
            domains[v].discard(u_val)
            if not domains[v]:
                return False
            # Re-add arcs
            for w in adj[v]:
                if w != u:
                    queue.append((w, v))
                    
    return True
