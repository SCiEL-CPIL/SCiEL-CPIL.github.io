def construct():
    """
    Constructs two disjoint K_{3,3} graphs on vertices 0-5 and 6-11.
    This yields a triangle-free graph with min degree 3 and distinct cycle lengths {4, 6}.
    Raw value = 2.
    """
    edges = []
    # K_{3,3} on 0-5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    # K_{3,3} on 6-11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    
    edges.sort()
    return [[int(u), int(v)] for u, v in edges]
