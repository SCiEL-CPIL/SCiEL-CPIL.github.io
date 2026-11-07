def construct():
    """
    Constructs a 3-regular bipartite graph on 12 vertices consisting of two disjoint K_{3,3} components.
    
    Properties:
    - Vertices: 0..11
    - Degrees: All vertices have degree 3.
    - Triangle-free: The graph is bipartite, so it contains no odd cycles, specifically no triangles.
    - Cycles: Each K_{3,3} component contains 4-cycles and 6-cycles.
    - Distinct Cycle Lengths: {4, 6}. Count = 2.
    
    This is optimal because:
    1. Girth must be at least 3.
    2. If girth is 3, we can use K_4 components. K_4 has cycles of length 3 and 4. Count = 2.
    3. If girth is 4 (bipartite), we must have 4-cycles. On 12 vertices with degree >= 3, 
       it is difficult to avoid 6-cycles. K_{3,3} has 4 and 6. Count = 2.
    4. It is highly unlikely that a graph exists with only 1 distinct cycle length 
       (e.g., all cycles length 4) with degree >= 3 and 12 vertices.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 (left) and 3, 4, 5 (right)
    left_1 = [0, 1, 2]
    right_1 = [3, 4, 5]
    for u in left_1:
        for v in right_1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 (left) and 9, 10, 11 (right)
    left_2 = [6, 7, 8]
    right_2 = [9, 10, 11]
    for u in left_2:
        for v in right_2:
            edges.append([u, v])

    # Ensure types are strictly Python int and list
    # The loop above already creates Python ints and lists.
    # Sorting for consistency is optional but good practice.
    # The task requires returning [u, v] lists.
    # No loops or duplicates.
    
    return edges
