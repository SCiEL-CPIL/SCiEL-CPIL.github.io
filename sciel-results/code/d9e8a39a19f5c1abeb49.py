def construct():
    """
    Construct K_{8,8}: complete bipartite graph on 16 vertices.
    Vertices 0-7 form part A, vertices 8-15 form part B.
    All edges between A and B are included.
    
    Properties:
    - 64 edges
    - Triangle-free (bipartite)
    - Diameter exactly 2
    - Every edge is essential: in a triangle-free graph, adjacent vertices
      share no common neighbor, so removing any edge (u,v) makes dist(u,v)=3
    """
    n = 16
    half = n // 2  # 8
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Ensure all values are Python int (not numpy types or bools)
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
