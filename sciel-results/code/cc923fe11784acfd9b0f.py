import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    
    # Initialize adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    added_edges = []
    
    # Strategy: Connect all even vertices to vertex 11 (which is odd).
    # This preserves the bipartite property (and thus triangle-free).
    # Vertex 11 becomes a hub with eccentricity 2, ensuring diameter <= 4.
    
    hub = 11
    even_vertices = [i for i in range(n) if i % 2 == 0]
    
    for u in even_vertices:
        if time.time() - t0 > deadline:
            break
        if hub in adj[u]:
            continue
        common = adj[u] & adj[hub]
        if len(common) > 0:
            continue
        adj[u].add(hub)
        adj[hub].add(u)
        edge_set.add((min(u, hub), max(u, hub)))
        added_edges.append([int(u), int(hub)])
        
    return added_edges
