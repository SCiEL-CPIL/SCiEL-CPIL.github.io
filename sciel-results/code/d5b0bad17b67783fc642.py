"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 16
    half = n // 2
    
    # Construct K_{8,8}
    # Partition 1: 0 to half-1
    # Partition 2: half to n-1
    # This is a triangle-free graph with diameter 2.
    # Removing any edge (u, v) with u in A, v in B:
    # dist(u, v) becomes 3 (via u->b'->a'->v) because N(u) and N(v) are disjoint.
    # Thus diameter increases to 3.
    # Edge count is 8 * 8 = 64, which is the maximum possible for a triangle-free graph on 16 vertices.
    
    edges = []
    for u in range(half):
        if time.time() > deadline:
            break
        for v in range(half, n):
            if time.time() > deadline:
                break
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
