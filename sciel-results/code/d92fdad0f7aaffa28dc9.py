import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 16
    # Try K_8,8 first
    edges = []
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
            
    if _is_valid(edges, n):
        return edges
        
    # Fallback to Star
    edges = []
    for i in range(1, n):
        edges.append([0, i])
        
    return edges

def _is_valid(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check Diameter 2
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist or max(dist) > 2:
            return False
            
    # Check Essential Edges
    for e in edges:
        u, v = e
        # Remove e
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Check Disconnection
        visited = [False] * n
        visited[0] = True
        q = [0]
        head = 0
        while head < len(q):
            x = q[head]
            head += 1
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = True
                    q.append(y)
        if not all(visited):
            adj[u].add(v)
            adj[v].add(u)
            continue # Disconnected, so essential
            
        # Check Diameter > 2
        diam = 0
        valid_diam = True
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = [s]
            head = 0
            while head < len(q):
                x = q[head]
                head += 1
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        q.append(y)
            if -1 in dist:
                valid_diam = False
                break
            if max(dist) > 2:
                valid_diam = False
                break
            if max(dist) > diam:
                diam = max(dist)
                
        adj[u].add(v)
        adj[v].add(u)
        
        if valid_diam and diam <= 2:
            return False # Edge not essential
            
    return True
