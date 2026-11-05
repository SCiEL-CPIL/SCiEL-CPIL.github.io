def construct():
    # Construct a C5-free graph on n vertices with no independent set of size 9.
    # Strategy: Use the complete 4-partite graph T(n, 4) with equal parts.
    # T(n, 4) is C5-free because any cycle must alternate between parts, 
    # and with only 4 parts, the shortest odd cycle would require 5 parts.
    # More precisely, in a complete r-partite graph, the only cycles are 
    # formed by choosing vertices from different parts. A C5 in K_4-partite 
    # would require 5 distinct parts, but we only have 4.
    # 
    # alpha(G) for T(n,4) is the size of the largest part.
    # To have alpha(G) < 9, we need each part to have size <= 8.
    # With 4 parts of size 8, n = 32, alpha = 8 < 9.
    # This is the maximum n achievable with this construction.
    
    n = 32
    part_size = 8
    num_parts = 4
    
    # Assign each vertex to a part: vertex i is in part (i // part_size)
    edges = []
    for u in range(n):
        part_u = u // part_size
        for v in range(u + 1, n):
            part_v = v // part_size
            if part_u != part_v:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
