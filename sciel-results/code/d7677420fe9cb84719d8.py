"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = 16
    # Partition vertices into two sets of 8
    U = list(range(0, 8))
    V = list(range(8, 16))
    
    edges = []
    for u in U:
        if time.time() > deadline:
            return edges
        for v in V:
            if time.time() > deadline:
                return edges
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
