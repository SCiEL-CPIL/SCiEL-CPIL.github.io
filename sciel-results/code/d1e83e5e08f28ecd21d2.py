import time

def construct():
    start = time.time()
    deadline = start + 55  # 55s budget to be safe

    n = 15
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Helper: compute degrees from edge set
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    # Helper: check if a given edge set is valid (no 3-edge path with equal-degree endpoints)
    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # For every edge (u, v), check all s in adj[u] and t in adj[v] with s != t
        # The path is s - u - v - t. Check deg[s] != deg[t]
        for u, v in edges:
            for s in adj[u]:
                for t in adj[v]:
                    if s != t:
                        if deg[s] == deg[t]:
                            return False
        return True

    # Helper: count edges
    def edge_count(edges):
        return len(edges)

    # Greedy insertion
    current_edges = set()
    deg = [0] * n
    adj = [set() for _ in range(n)]

    # Try adding edges in a smart order: prefer high-degree connections first
    # Actually, let's just try all edges and add if valid
    # For speed, we maintain adjacency and degrees incrementally

    # Fast validity check for adding one edge (u, v)
    # A path s-u-v-t exists if s in adj[u] and t in adj[v] and s != t
    # We need deg[s] != deg[t] for all such pairs.
    # Also need to check paths where (u,v) is not the middle edge:
    #   Path s-u-v-t where (u,v) is first edge: s-u-v-w? No, 3-edge path is 4 vertices.
    #   The constraint is on EVERY 3-edge path. Adding (u,v) creates new 3-edge paths
    #   where (u,v) is one of the three edges.
    #
    # Cases for 3-edge path involving new edge (u,v):
    # 1. (u,v) is middle edge: s-u-v-t, s in adj[u], t in adj[v], s != t, s != v, t != u
    #    Need deg[s] != deg[t]
    # 2. (u,v) is first edge: u-v-x-y, x in adj[v] (x != u), y in adj[x] (y != v)
    #    Need deg[u] != deg[y]
    # 3. (u,v) is last edge: y-x-v-u, x in adj[v] (x != u), y in adj[x] (y != v)
    #    Need deg[y] != deg[v]
    #
    # Also paths where (u,v) is first or last but the other edges are existing.
    #
    # This is complex. Let's use a simpler but still efficient check.

    def can_add_edge(edges_set, deg, adj, u, v):
        # Check all 3-edge paths that include (u,v)
        # We simulate adding (u,v) temporarily
        
        # Case 1: (u,v) is middle edge: s - u - v - t
        # s in adj[u], t in adj[v], s != t, s != v, t != u
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # Case 2: (u,v) is first edge: u - v - x - y
        # x in adj[v] (x != u), y in adj[x] (y != v)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == v:
                    continue
                if deg[u] == deg[y]:
                    return False
        
        # Case 3: (u,v) is last edge: y - x - v - u
        # x in adj[v] (x != u), y in adj[x] (y != v)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == v:
                    continue
                if deg[y] == deg[v]:
                    return False
        
        # Case 4: (u,v) is first edge: u - v - x - y where x in adj[v], y in adj[x]
        # Wait, I already did case 2. Let me also check symmetric cases.
        # Actually the path is undirected, so (u,v) as first edge covers u-v-x-y
        # and (u,v) as last edge covers y-x-v-u which is the same path reversed.
        # So cases 2 and 3 are symmetric and I need to check both directions.
        # But I already checked both. Let me also check:
        
        # Case 5: (u,v) is first edge: u - v - x - y where x in adj[v]
        # Already covered in case 2.
        
        # Case 6: (u,v) is last edge: y - x - v - u
        # Already covered in case 3.
        
        # What about paths where (u,v) is the second edge but not middle?
        # In a 3-edge path, the edges are e1, e2, e3. 
        # If (u,v) is e1: u-v-x-y
        # If (u,v) is e2: s-u-v-t
        # If (u,v) is e3: y-x-v-u
        # I've covered all three.
        
        return True

    # Start with no edges, greedily add
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if (u, v) in current_edges:
            continue
        if can_add_edge(current_edges, deg, adj, u, v):
            current_edges.add((u, v))
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

    # Convert to list format
    best_edges = [[u, v] for u, v in current_edges]
    best_count = len(best_edges)

    # Try local search: drop one edge, try to add multiple
    # This is complex, so let's try a simpler improvement
    # Try dropping each edge and see if we can add more

    for drop_idx in range(len(best_edges)):
        if time.time() > deadline:
            break
        
        trial_edges = list(current_edges)
        drop_edge = (min(best_edges[drop_idx][0], best_edges[drop_idx][1]), 
                     max(best_edges[drop_idx][0], best_edges[drop_idx][1]))
        
        if drop_edge in trial_edges:
            trial_edges.remove(drop_edge)
        
        # Recompute degrees and adjacency for trial
        trial_deg = [0] * n
        trial_adj = [set() for _ in range(n)]
        for u, v in trial_edges:
            trial_deg[u] += 1
            trial_deg[v] += 1
            trial_adj[u].add(v)
            trial_adj[v].add(u)
        
        # Try adding edges that were not in trial
        added = 0
        for u, v in all_pairs:
            if (u, v) in trial_edges:
                continue
            if can_add_edge(trial_edges, trial_deg, trial_adj, u, v):
                trial_edges.append((u, v))
                trial_deg[u] += 1
                trial_deg[v] += 1
                trial_adj[u].add(v)
                trial_adj[v].add(u)
                added += 1
        
        if added > 1:  # Net gain of at least 1 edge
            current_edges = set(trial_edges)
            best_count = len(current_edges)
            deg = trial_deg
            adj = trial_adj

    # Final validation
    final_edges = [[u, v] for u, v in current_edges]
    if not is_valid(final_edges):
        # Fallback to bipartite
        final_edges = [[i, j] for i in range(7) for j in range(7, 15)]

    # Ensure types are correct
    result = []
    for e in final_edges:
        result.append([int(e[0]), int(e[1])])

    return result
