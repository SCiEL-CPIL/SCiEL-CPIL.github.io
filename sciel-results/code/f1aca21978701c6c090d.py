def construct():
    import itertools
    
    # Vertices are 4-subsets of {0, 1, 2, 3, 4, 5, 6}
    base_set = range(7)
    vertices = list(itertools.combinations(base_set, 4))
    n = len(vertices) # 35
    
    # Map vertex (tuple) to index
    index_map = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a = set(vertices[i])
            b = set(vertices[j])
            inter_size = len(a.intersection(b))
            # Adjacent in G_ramsey if intersection is NOT 2
            # Intersections possible: 0, 1, 3 (4 is i=j)
            if inter_size != 2:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
