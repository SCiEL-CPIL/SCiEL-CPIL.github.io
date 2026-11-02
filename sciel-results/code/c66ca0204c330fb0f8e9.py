import hashlib

def construct(k, g):
    q = k
    candidates = []
    for d in range(3, 8):
        n = 2 * (q ** d)
        if n <= 10000:
            candidates.append(n)
    
    for n in candidates:
        # Build adjacency list using 5 hash-based perfect matchings
        adj = [[] for _ in range(n)]
        
        for i in range(k):
            # Compute hash for each vertex for this matching
            hashes = []
            for v in range(n):
                h = hashlib.sha256(f"{i}:{v}".encode()).digest()
                hashes.append((int.from_bytes(h[:8], 'big'), v))
            
            # Sort by hash value
            hashes.sort()
            
            # Pair consecutive vertices
            for j in range(0, n, 2):
                u = hashes[j][1]
                v = hashes[j+1][1]
                adj[u].append(v)
                adj[v].append(u)
        
        # Verify girth using BFS
        girth = float('inf')
        
        for start_v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start_v] = 0
            queue = [start_v]
            found_short = False
            
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                        if girth < g:
                            found_short = True
                            break
                if found_short:
                    break
            if found_short:
                break
        
        if girth >= g:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    raise ValueError("Construction failed")
