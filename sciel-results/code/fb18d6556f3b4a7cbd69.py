def construct():
    """
    Constructs two disjoint K_{3,3} graphs on vertices 0-5 and 6-11.
    
    Properties:
    - 12 vertices (0..11)
    - Undirected, simple
    - Min degree: 3 (each vertex in K_{3,3} has degree 3)
    - Triangle-free: K_{3,3} is bipartite
    - Cycle lengths: K_{3,3} has cycles of length 4 and 6 only.
      - 4-cycles exist (e.g., 0-3-1-4-0)
      - 6-cycles exist (e.g., 0-3-1-4-2-5-0)
      - No cycles of length > 6 because max vertices in a partition is 3.
    - Distinct cycle lengths: {4, 6} -> Count = 2.
    
    This is the minimum possible count for triangle-free graphs with min degree >= 3.
    """
    edges = []
    # K_{3,3} on 0-5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # K_{3,3} on 6-11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    # Sort edges for deterministic output
    edges.sort()
    
    return edges
