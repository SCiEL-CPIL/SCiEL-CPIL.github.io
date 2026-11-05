def construct():
    edges = []
    n = 16
    for u in range(n):
        r_u, c_u = divmod(u, 4)
        for v in range(u + 1, n):
            r_v, c_v = divmod(v, 4)
            if r_u != r_v and c_u != c_v:
                edges.append([u, v])
    return edges
