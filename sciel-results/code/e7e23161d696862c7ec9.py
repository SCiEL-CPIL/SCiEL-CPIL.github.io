"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 59  # Safety margin for 60s limit
    
    n = P['n']
    half = n // 2
    
    # Partitions for Complete Bipartite Graph K_{n/2, n/2}
    # This graph is triangle-free (max edges per Mantel's Theorem)
    # and has diameter 2.
    # Every edge is essential because removing any edge (u,v) with u in A, v in B
    # increases dist(u,v) to 3 (since no path of length 2 exists in bipartite graphs 
    # between different partitions), thus increasing diameter to 3.
    
    edges = []
    for u in range(half):
        if time.time() > deadline:
            break
        for v in range(half, n):
            if time.time() > deadline:
                break
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
