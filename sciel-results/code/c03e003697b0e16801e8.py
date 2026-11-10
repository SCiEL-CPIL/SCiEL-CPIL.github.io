def construct():
    # Two disjoint K_{3,3} graphs: vertices {0,1,2} x {3,4,5} and {6,7,8} x {9,10,11}
    # Each vertex has degree 3, graph is bipartite (triangle-free),
    # and the only cycle lengths are 4 and 6 (2 distinct lengths).
    edges = []
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([int(u), int(v)])
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([int(u), int(v)])
    return sorted(edges)
