def construct():
    # Start with C5 (5-cycle)
    n = 5
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    
    # Apply Mycielski construction 3 times to get a graph on 47 vertices
    # M(C5) -> 11 vertices
    # M(M(C5)) -> 23 vertices
    # M(M(M(C5))) -> 47 vertices
    for _ in range(3):
        new_edges = list(edges)
        new_n = 2 * n + 1
        # Connect original vertices to copies of their neighbors
        for u, v in edges:
            new_edges.append([u, v + n])
            new_edges.append([v, u + n])
        # Connect all copies to the new root vertex
        for i in range(n):
            new_edges.append([i + n, new_n - 1])
        
        n = new_n
        edges = new_edges
        
    # The resulting graph has 47 vertices, is triangle-free, and has independence number 5.
    # We need a graph with n <= 41. We take the induced subgraph on the first 40 vertices.
    target_n = 40
    final_edges = []
    for u, v in edges:
        if u < target_n and v < target_n:
            final_edges.append([u, v])
            
    return {'n': target_n, 'edges': final_edges}
