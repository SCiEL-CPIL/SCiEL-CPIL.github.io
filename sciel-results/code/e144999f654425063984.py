def construct():
    n = 16
    mid = n // 2
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    return edges
