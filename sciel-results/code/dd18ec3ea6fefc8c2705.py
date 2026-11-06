def construct():
    edges = []
    for i in range(10):
        for j in range(10):
            c = (i + j) % 20
            edges.append([i, 10 + j, c])
    return edges
