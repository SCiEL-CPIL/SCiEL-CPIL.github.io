def construct():
    # Two disjoint K_{3,3} components on vertices 0..5 and 6..11.
    # K_{3,3} is bipartite, triangle-free, min degree 3, and has cycles of
    # length 4 and 6 only. Overall distinct cycle lengths = {4, 6}, raw = 2.
    # This is optimal for the design family of bipartite min-degree-3 graphs.
    
    edges = []
    for offset in (0, 6):
        part_a = [offset, offset + 1, offset + 2]
        part_b = [offset + 3, offset + 4, offset + 5]
        for u in part_a:
            for v in part_b:
                edges.append([u, v])
    
    # Quick structural verification (always passes for the construction above)
    n = 12
    edge_set = set()
    for e in edges:
        u, v = e
        if u == v or not (0 <= u < n and 0 <= v < n):
            return []
        if (u, v) in edge_set or (v, u) in edge_set:
            return []
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    return edges
