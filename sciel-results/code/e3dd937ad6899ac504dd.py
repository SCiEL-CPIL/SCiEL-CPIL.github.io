def construct():
    # 8 disjoint copies of K4
    # n = 32
    # alpha = 8
    # C5-free
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': 32, 'edges': edges}
