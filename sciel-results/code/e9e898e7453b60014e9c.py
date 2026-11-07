def construct():
    """
    Construct a 12-vertex undirected simple graph with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Contains at least one cycle
    - Minimizes the number of distinct simple cycle lengths
    
    The optimal solution for 12 vertices with min degree 3 and triangle-free constraint
    is two disjoint K_{3,3} components.
    
    Each K_{3,3} component has:
    - 6 vertices
    - Min degree 3
    - Cycles of length 4 and 6 only.
    
    Since the components are disjoint, the set of distinct cycle lengths for the whole graph
    is the union of the sets from each component, which is {4, 6}.
    The number of distinct cycle lengths is 2.
    
    It is theoretically impossible to have only 1 distinct cycle length (e.g., all 4s)
    with minimum degree 3 on 12 vertices, as any connected component with min degree 3
    and girth 4 must contain a cycle of length other than 4 (typically 6) due to
    combinatorial constraints on bipartite graphs (e.g., existence of 6-cycles in
    K_{3,3} or similar structures).
    
    Therefore, 2 is the minimum possible raw value.
    """
    import time
    start_time = time.time()
    
    # Two disjoint K_{3,3} components
    # Component 1: Vertices 0,1,2 (Part A) and 3,4,5 (Part B)
    # Component 2: Vertices 6,7,8 (Part A) and 9,10,11 (Part B)
    
    edges = []
    
    # Generate edges for first K_{3,3}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Generate edges for second K_{3,3}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Sort edges for deterministic output
    edges.sort()
    
    # Ensure types are strictly Python int and list
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
