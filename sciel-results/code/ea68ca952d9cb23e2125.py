import time

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Grötzsch graph: Mycielski of C5
    # C5: vertices 0-4
    # Mycielski: 0-4 (original), 5-9 (copies), 10 (new vertex)
    
    edges = []
    
    # C5 edges
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Copy edges: for each edge (u,v) in C5, add (u, 5+v) and (v, 5+u)
    c5_edges = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in c5_edges:
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    
    # Connect all copies to vertex 10
    for i in range(5):
        edges.append([5 + i, 10])
    
    # Now we have 11 vertices (0-10) with 20 edges, triangle-free, 4-chromatic
    # Vertices 11-15 are isolated
    # Greedily add edges while maintaining triangle-freeness
    # Since we're only adding edges to a 4-chromatic graph, chromatic number stays >= 4
    
    n = 16
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: check if adding edge (u,v) creates a triangle
    def creates_triangle(u, v):
        for w in adj[u]:
            if w in adj[v]:
                return True
        return False
    
    # Candidate edges: all possible edges not yet present
    candidate_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                candidate_edges.append((i, j))
    
    # Sort by degree sum (prefer connecting high-degree vertices)
    def edge_priority(e):
        u, v = e
        return -(len(adj[u]) + len(adj[v]))
    
    candidate_edges.sort(key=edge_priority)
    
    # Greedily add edges that don't create triangles
    for u, v in candidate_edges:
        if time.time() > deadline - 5:
            break
        
        if (u, v) in edge_set:
            continue
        if creates_triangle(u, v):
            continue
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edges.append([u, v])
    
    return edges
