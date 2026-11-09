import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Base solution: 8 disjoint K4s
    # n = 32
    # Edges: for each block b in 0..7, vertices 4b..4b+3 form a K4
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Return the known best solution
    return {'n': n, 'edges': edges}
