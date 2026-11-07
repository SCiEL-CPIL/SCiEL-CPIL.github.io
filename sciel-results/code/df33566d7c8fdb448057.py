"""Dense diameter-2 graph on 16 vertices where every edge is critical."""
# EVOLVE-BLOCK-START

import time

def construct():
    # Deadline guard: return early if we're running out of time
    start_time = time.time()
    deadline = start_time + 60.0  # 60s wall-clock limit
    
    # The optimal structure is K_{8,8}: triangle-free, diameter 2,
    # every edge critical (removing edge (u,v) makes dist(u,v) = 3).
    # By Turán's theorem, no triangle-free graph on 16 vertices can
    # have more than 8*8 = 64 edges, so this is provably optimal.
    left = [0, 1, 2, 3, 4, 5, 6, 7]
    right = [8, 9, 10, 11, 12, 13, 14, 15]

    edges = []
    for a in left:
        for b in right:
            edges.append([a, b])
        # Check deadline periodically (cheap check)
        if time.time() > deadline - 1.0:
            # Return what we have so far (should be complete by now)
            return edges

    return edges

# EVOLVE-BLOCK-END
