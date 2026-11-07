"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    deadline = time.time() + 50
    n = P['n']
    # Build K_{8,8}: partition into two sets of 8, connect all cross pairs
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
