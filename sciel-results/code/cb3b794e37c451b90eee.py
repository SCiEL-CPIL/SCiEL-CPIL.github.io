def construct():
    n = 16
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if u % 2 != v % 2:
                edges.append([u, v])
    return edges
