"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    
    # The problem requires a graph with diameter 2 where every edge is critical.
    # This implies the graph must be triangle-free.
    # By Mantel's Theorem, the max edges in a triangle-free graph on n vertices is floor(n^2/4).
    # For n=16, max edges = 64, achieved by K_{8,8}.
    # K_{8,8} has diameter 2.
    # Removing any edge (u,v) from K_{8,8} makes dist(u,v)=3, so diameter becomes 3.
    # Thus, all edges are critical.
    
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
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
