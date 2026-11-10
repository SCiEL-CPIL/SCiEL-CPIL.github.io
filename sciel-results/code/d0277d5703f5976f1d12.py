def _greedy_cycle_decompose(edges_set, all_cycles, deadline):
    remaining = set(edges_set)
    pieces = []
    adj = _build_adj(remaining)
    
    # Sort cycles by length descending
    sorted_cycles = sorted(all_cycles, key=len, reverse=True)
    
    # We need to check if a cycle is still "available" (all its edges are in remaining)
    # This is O(len(cycle)) per check.
    # To speed up, we can just iterate and check.
    
    while remaining and time.time() < deadline:
        found = False
        for cyc in sorted_cycles:
            # Check if all edges in cyc are in remaining
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1)%len(cyc)]
                e = tuple(sorted((u,v)))
                if e not in remaining:
                    valid = False
                    break
            if valid:
                pieces.append(cyc)
                # Remove edges
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    e = tuple(sorted((u,v)))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]: del adj[u]
                    if not adj[v]: del adj[v]
                found = True
                break
        
        if not found:
            # No valid cycle from list, pick an edge
            if remaining:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]: del adj[u]
                if not adj[v]: del adj[v]
            else:
                break
                
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces
