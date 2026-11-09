def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # This satisfies alpha < 9 and maximizes n for this construction.
    n = 32
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges.append([u, v])
    return {'n': n, 'edges': edges}
