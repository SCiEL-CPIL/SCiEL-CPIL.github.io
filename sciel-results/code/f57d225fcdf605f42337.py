"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n = 16
    # Direct construction of K_{8,8} which has 64 edges, diameter 2, and is edge-critical
    # for diameter 2. This is the theoretical maximum by Mantel's theorem (triangle-free
    # extremal graph) and every edge is critical since removing any edge (u,v) makes
    # dist(u,v) = 3 (via any other vertex in the opposite partition).
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
