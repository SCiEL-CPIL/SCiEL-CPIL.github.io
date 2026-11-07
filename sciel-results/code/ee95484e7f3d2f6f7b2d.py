# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 59  # Safety margin

    half = n // 2
    edges = []
    
    # K_{8,8} is the maximum triangle-free graph on 16 vertices.
    # Triangle-free is necessary for all edges to be critical in a diameter-2 graph.
    for u in range(half):
        for v in range(half, n):
            if time.time() > deadline:
                return edges
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
