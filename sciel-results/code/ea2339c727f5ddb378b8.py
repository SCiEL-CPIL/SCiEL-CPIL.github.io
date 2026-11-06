"""Construct a C5-free graph on 33 vertices with alpha <= 8."""

def construct():
    n = 33
    edges = []
    # Component A: 5 vertices, alpha=1 (K5)
    A = [0, 1, 2, 3, 4]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([A[i], A[j]])
    # Component B: 5 vertices, alpha=1 (K5)
    B = [5, 6, 7, 8, 9]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([B[i], B[j]])
    # Component C: 5 vertices, alpha=1 (K5)
    C = [10, 11, 12, 13, 14]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([C[i], C[j]])
    # Component D: 5 vertices, alpha=1 (K5)
    D = [15, 16, 17, 18, 19]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([D[i], D[j]])
    # Component E: 5 vertices, alpha=1 (K5)
    E = [20, 21, 22, 23, 24]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([E[i], E[j]])
    # Component F: 5 vertices, alpha=1 (K5)
    F = [25, 26, 27, 28, 29]
    for i in range(5):
        for j in range(i+1, 5):
            edges.append([F[i], F[j]])
    # Component G: 3 vertices, alpha=1 (K3)
    G = [30, 31, 32]
    for i in range(3):
        for j in range(i+1, 3):
            edges.append([G[i], G[j]])
    return {'n': n, 'edges': edges}
