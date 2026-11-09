import time

def construct():
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
