def construct():
    n = 16
    half = n // 2
    edges = []
    for i in range(half):
        for j in range(half, n):
            edges.append([i, j])
    return edges
