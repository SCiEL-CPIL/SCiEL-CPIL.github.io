"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # K_{8,8} is optimal: triangle-free graphs max at floor(n^2/4) = 64 edges (Mantel's theorem).
    # In K_{8,8}, every edge (u,v) has u,v with no common neighbor, making every edge critical.
    # Diameter is exactly 2 (any two vertices in different partitions have distance 1,
    # any two in same partition have distance 2 via any vertex in the other partition).
    
    edges = []
    half = n // 2
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
