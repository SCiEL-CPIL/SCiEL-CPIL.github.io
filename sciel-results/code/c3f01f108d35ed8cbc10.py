from collections import deque

def construct(k, g):
    # Search for Generalized Petersen Graph G(n, k_gpg) with girth >= g
    # Vertices: 0..2n-1
    # Edges: (i, i+1)%n, (n+i, n+(i+k_gpg))%n, (i, n+i)
    
    best_n = None
    best_edges = None
    
    # n must be at least g/2 + something?
    # Girth of G(n, k) is bounded.
    # Let's search n up to 200.
    
    for n in range(10, 200):
        # k_gpg must be coprime to n for connectivity?
        # Actually, for 3-regular, it doesn't strictly need to be connected?
        # "Simple graph... every vertex degree k".
        # Usually cages are connected.
        # If not connected, we can just take the component?
        # But "n" is the number of vertices in the graph returned.
        # If disconnected, we return all vertices.
        # But girth is min over components.
        # So we need all components to have girth >= g.
        
        for k_gpg in range(1, n // 2 + 1):
            # Build graph
            adj = [[] for _ in range(2 * n)]
            edges = set()
            
            def add_edge(u, v):
                if u == v: continue
                if (u, v) in edges or (v, u) in edges: continue
                adj[u].append(v)
                adj[v].append(u)
                edges.add((u, v))
                edges.add((v, u))
                
            for i in range(n):
                add_edge(i, (i + 1) % n)
                add_edge(n + i, n + (i + k_gpg) % n)
                add_edge(i, n + i)
                
            # Check if 3-regular
            valid = True
            for i in range(2 * n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check girth
            girth = get_girth(adj, 2 * n, g)
            if girth >= g:
                # This is a candidate
                if best_n is None or 2 * n < best_n:
                    best_n = 2 * n
                    best_edges = [(min(u, v), max(u, v)) for u, v in edges if u < v]
                    
    if best_n is None:
        # Fallback to random search or seed
        # For now, return a large random one or seed
        pass
        
    return best_n, best_edges

def get_girth(adj, n, limit):
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] >= limit // 2:
                        continue
                    q.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < limit:
                        return cycle_len
    return limit
