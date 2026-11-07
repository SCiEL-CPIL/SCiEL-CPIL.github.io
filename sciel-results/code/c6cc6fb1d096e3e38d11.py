"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    import heapq
    from collections import deque
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Candidate edges: only between even and odd (bipartite), not already present
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def bfs_distances(adj):
        """Compute all-pairs shortest path distances."""
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            dq = deque()
            dist[src][src] = 0
            dq.append(src)
            while dq:
                node = dq.popleft()
                for nb in adj[node]:
                    if dist[src][nb] == -1:
                        dist[src][nb] = dist[src][node] + 1
                        dq.append(nb)
        return dist
    
    def max_dist(dist):
        m = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > m:
                    m = dist[i][j]
        return m
    
    added = []
    added_set = set()
    
    # Greedy: repeatedly add the edge that most reduces max distance
    while True:
        dist = bfs_distances(adj)
        current_max = max_dist(dist)
        if current_max <= 4:
            break
        
        best_edge = None
        best_new_max = current_max
        best_reduction = 0
        
        for (u, v) in candidates:
            if (u, v) in added_set:
                continue
            # Tentatively add edge
            adj[u].append(v)
            adj[v].append(u)
            
            dist2 = bfs_distances(adj)
            new_max = max_dist(dist2)
            
            reduction = current_max - new_max
            if reduction > best_reduction:
                best_reduction = reduction
                best_new_max = new_max
                best_edge = (u, v)
            
            # Remove tentative edge
            adj[u].remove(v)
            adj[v].remove(u)
        
        if best_edge is None or best_reduction <= 0:
            # No improvement possible with remaining candidates
            break
        
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        added_set.add((u, v))
        added.append([u, v])
    
    # Convert to proper Python int lists
    result = []
    for u, v in added:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END

