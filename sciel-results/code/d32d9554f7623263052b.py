"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    # K_{8,8} is the unique triangle-free graph on 16 vertices with maximum edges (64).
    # By Mantel's theorem, max edges in triangle-free graph on n vertices is floor(n^2/4) = 64.
    # Triangle-free is NECESSARY: if u,v,w form a triangle, removing edge (u,v) leaves
    # path u-w-v of length 2, so (u,v) is not critical.
    # In K_{8,8}, removing any edge (u,v) with u in part A, v in part B makes
    # dist(u,v) = 3 (via any other vertex), so diameter becomes 3 > 2. Every edge is critical.
    # 64 is the theoretical maximum; this is optimal.
    
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([int(a), int(b)])
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Ensure all values are Python ints (not numpy or bool)
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
