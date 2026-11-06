import time

def construct():
    # Andrasfai graph And(9) on 27 vertices
    # Triangle-free, alpha = 9
    n = 27
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
