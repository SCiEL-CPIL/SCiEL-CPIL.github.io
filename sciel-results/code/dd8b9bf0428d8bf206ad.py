"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard to ensure we finish well within the 60s/90s limit
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    
    # The problem requires a graph with diameter 2 where every edge is critical.
    # 1. If a graph has a triangle, removing an edge from the triangle leaves a path of length 2,
    #    so the diameter remains 2. Thus, for all edges to be critical, the graph must be triangle-free.
    # 2. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices
    #    is floor(n^2 / 4). For n=16, this is 64.
    # 3. The complete bipartite graph K_{8,8} achieves this bound.
    # 4. In K_{8,8}, the diameter is 2. Removing any edge (u,v) disconnects u and v directly,
    #    and since N(u) and N(v) are disjoint (other than each other, which is removed),
    #    the new distance is 3. Thus, diameter increases to 3.
    
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        if time.time() > deadline:
            break
        for v in B:
            if time.time() > deadline:
                break
            # Ensure types are exactly Python int and list
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
