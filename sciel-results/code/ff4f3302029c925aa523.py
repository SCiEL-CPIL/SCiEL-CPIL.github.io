def construct():
    n = 256
    # 1. Build conflict graph
    # adj[i] is a set of nodes conflicting with i
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    # 2. Greedy Vertex Cover (Max-Degree Heuristic)
    # We build a set 'cover' of nodes to exclude.
    # The independent set will be all nodes NOT in cover.
    cover = set()
    remaining_edges = set()
    
    # Initialize remaining edges
    for u in range(1, n + 1):
        for v in adj[u]:
            if u < v:
                remaining_edges.add((u, v))
                
    # To efficiently find max degree, we can just iterate over remaining nodes
    # Since n is small (256), O(n * |E|) is fine.
    active_nodes = set(range(1, n + 1))
    
    while remaining_edges:
        # Find node with max degree in the subgraph induced by remaining_edges
        max_deg_node = None
        max_deg = -1
        
        for u in active_nodes:
            # Count edges in remaining_edges incident to u
            deg = 0
            for e in remaining_edges:
                if e[0] == u or e[1] == u:
                    deg += 1
            
            if deg > max_deg:
                max_deg = deg
                max_deg_node = u
                
        if max_deg_node is None:
            break
            
        # Add to cover
        cover.add(max_deg_node)
        active_nodes.discard(max_deg_node)
        
        # Remove all edges incident to max_deg_node
        # Create a new set of remaining edges
        new_remaining = set()
        for e in remaining_edges:
            if e[0] != max_deg_node and e[1] != max_deg_node:
                new_remaining.add(e)
        remaining_edges = new_remaining
        
    # The independent set is all nodes not in cover
    independent_set = set(range(1, n + 1)) - cover
    
    # 3. Local Search Refinement
    # Check if any node in cover can be moved to independent_set
    # A node u in cover can be moved to IS if all its neighbors are in cover.
    # Because then it has no neighbors in IS.
    improved = True
    while improved:
        improved = False
        for u in list(cover):
            # Check if all neighbors of u are in cover
            can_move = True
            for v in adj[u]:
                if v not in cover:
                    can_move = False
                    break
            if can_move:
                cover.discard(u)
                improved = True
                # Restart loop as cover changed
                break
                
    return list(independent_set - cover) # Wait, independent_set was calculated before refinement.
    # Actually, independent_set = All - cover.
    # So just return list(set(range(1, n+1)) - cover)
