"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # Candidate edges: all pairs not already in base, and not creating a triangle
    # A pair (u, v) creates a triangle if there exists w such that (u,w) and (w,v) are both in the current graph
    # Since we start with a path, the only triangles would be formed by adding an edge (u,v) where dist_path(u,v) == 2
    # i.e., u and v have a common neighbor in the path.
    
    # Let's build the adjacency for the base path
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges that don't create triangles with the base path
    # (u, v) is valid if no w is in adj[u] ∩ adj[v]
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            # Check triangle-free condition with base
            if adj[u] & adj[v]:
                continue
            candidates.append((u, v))
    
    # Now we need to select a minimum number of these candidates such that:
    # 1. The resulting graph is connected (it already is, since base is a path)
    # 2. The diameter is at most 4
    # 3. The graph remains triangle-free (need to check with added edges too)
    
    # Since the base is already connected, we just need to reduce the diameter to <= 4.
    # The current diameter is 23 (path from 0 to 23).
    # We need to add edges to shorten the longest shortest path to at most 4.
    
    # Greedy approach: repeatedly find the pair of vertices with maximum distance,
    # and add the best candidate edge that most reduces the diameter.
    
    from collections import deque
    
    def bfs_distances(adj_list, n):
        """Compute all-pairs shortest path distances using BFS from each node."""
        dist = [[-1] * n for _ in range(n)]
        for s in range(n):
            dist[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[s][v] == -1:
                        dist[s][v] = dist[s][u] + 1
                        q.append(v)
        return dist
    
    def get_diameter(dist):
        return max(max(row) for row in dist)
    
    def is_triangle_free(adj_list, n, edge_set):
        """Check if the graph is triangle-free."""
        for u in range(n):
            for v in adj_list[u]:
                if v <= u:
                    continue
                for w in adj_list[u]:
                    if w > v and w in adj_list[v]:
                        return False
        return True
    
    # Start with the base graph
    current_adj = [list(adj[i]) for i in range(n)]
    current_edges = set(base_edges)
    added_edges = []
    
    deadline = time.time() + 80  # 80 second deadline, leave 10s margin
    
    # Compute initial distances
    dist = bfs_distances(current_adj, n)
    diam = get_diameter(dist)
    
    # Greedy: while diameter > 4, find the pair with max distance and try to add an edge
    while diam > 4:
        if time.time() > deadline:
            break
        
        # Find all pairs with distance == diam
        max_pairs = []
        for u in range(n):
            for v in range(u + 1, n):
                if dist[u][v] == diam:
                    max_pairs.append((u, v))
        
        if not max_pairs:
            break
        
        # For each candidate edge, compute how much it would reduce the diameter
        # We'll try each candidate and pick the one that gives the best diameter reduction
        best_edge = None
        best_diam = diam + 1
        best_dist = None
        
        for (u, v) in candidates:
            if (u, v) in current_edges:
                continue
            
            # Check triangle-free condition with current graph
            if set(current_adj[u]) & set(current_adj[v]):
                continue
            
            # Tentatively add the edge and compute new diameter
            new_adj = [list(row) for row in current_adj]
            new_adj[u].append(v)
            new_adj[v].append(u)
            
            new_dist = bfs_distances(new_adj, n)
            new_diam = get_diameter(new_dist)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                best_dist = new_dist
        
        if best_edge is None:
            break
        
        # Add the best edge
        u, v = best_edge
        current_adj[u].append(v)
        current_adj[v].append(u)
        current_edges.add((u, v))
        added_edges.append([u, v])
        dist = best_dist
        diam = best_diam
    
    # Verify the result is valid
    # Check triangle-free
    if not is_triangle_free(current_adj, n, current_edges):
        # Fallback: return empty or a known valid solution
        # Let's try a simpler heuristic
        return []
    
    # Check diameter
    final_dist = bfs_distances(current_adj, n)
    final_diam = get_diameter(final_dist)
    if final_diam > 4:
        return []
    
    # Check connectivity (should be fine since base is a path)
    # Check no duplicate edges
    for e in added_edges:
        if e[0] > e[1]:
            e[0], e[1] = e[1], e[0]
        if e[0] == e[1]:
            return []
    
    # Ensure all are Python ints
    result = []
    for u, v in added_edges:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
