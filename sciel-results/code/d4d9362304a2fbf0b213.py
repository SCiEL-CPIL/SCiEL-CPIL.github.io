import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    n = 16
    # Construct K_8,8 which is the maximum edge count for a triangle-free graph on 16 vertices.
    # Triangle-free + Diameter 2 implies all edges are essential.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # Check time
    if time.time() > deadline:
        return edges

    # Verify to ensure correctness
    if not _verify(edges, n):
        # Fallback to star graph if verification fails (should not happen)
        edges = [[0, v] for v in range(1, n)]
        
    # Ensure types are correct
    # edges is list of lists of ints.
    
    return edges

def _verify(edges, n):
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
        if -1 in dist:
            return False
        if max(dist) > 2:
            return False
            
    # Check Triangle-Free (Essential Edges)
    for u in range(n):
        neighbors = adj[u]
        for v in neighbors:
            # If any neighbor of v is also a neighbor of u, a triangle exists
            if adj[v] & neighbors:
                return False
                
    return True
