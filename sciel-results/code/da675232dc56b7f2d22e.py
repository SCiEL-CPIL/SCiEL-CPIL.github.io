def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Design: Two disjoint copies of K_{3,3}.
    Component 1: Vertices 0-5. Partitions {0,1,2} and {3,4,5}.
    Component 2: Vertices 6-11. Partitions {6,7,8} and {9,10,11}.
    
    Properties:
    - Bipartite => No odd cycles => Girth >= 4.
    - K_{3,3} has cycles of length 4 and 6.
    - Min degree is 3.
    - Distinct cycle lengths: {4, 6}. Count = 2.
    """
    
    edges = []
    
    # First K_{3,3} component using vertices 0..5
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} component using vertices 6..11
    # Partition A: {6, 7, 8}, Partition B: {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Validation to ensure correctness before returning
    # 1. Check for duplicates and loops
    edge_set = set()
    for e in edges:
        u, v = e
        if u == v:
            raise ValueError("Loop detected")
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            raise ValueError("Duplicate edge detected")
        edge_set.add((u, v))
        
    # 2. Check degrees
    degrees = [0] * 12
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
        
    for i in range(12):
        if degrees[i] < 3:
            raise ValueError(f"Vertex {i} has degree {degrees[i]} < 3")
            
    # 3. Check for triangles (girth >= 4)
    # Since the graph is bipartite by construction, there are no triangles.
    # But let's verify bipartiteness just in case.
    # We can check if the graph is bipartite using BFS/DFS.
    # However, the construction is explicitly bipartite.
    # We can also verify no 3-cycles by checking if any edge (u,v) has a common neighbor.
    # Adjacency list
    adj = {i: set() for i in range(12)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for u in range(12):
        for v in adj[u]:
            if u < v: # Check each edge once
                # Common neighbors of u and v form a triangle
                common = adj[u] & adj[v]
                if common:
                    raise ValueError(f"Triangle detected involving edge ({u}, {v})")
                    
    # 4. Check that it contains a cycle
    # K_{3,3} definitely contains cycles.
    
    # Return edges as list of lists of ints
    return edges
