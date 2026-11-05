def construct():
    n = 16
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor_val = u ^ v
            dist = bin(xor_val).count('1')
            if dist == 1 or dist == 4:
                edges.append([u, v])
    return edges
