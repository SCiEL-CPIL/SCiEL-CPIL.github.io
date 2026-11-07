def construct():
    n = 16
    part_a = list(range(8))
    part_b = list(range(8, 16))
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
    return edges
