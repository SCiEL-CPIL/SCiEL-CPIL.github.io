"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_DEADLINE_FRACTION = 0.85
_START = time.time()

def _deadline_reached():
    return (time.time() - _START) > 90.0 * _DEADLINE_FRACTION

def construct():
    # K_{8,8}: complete bipartite graph is the unique extremal triangle-free
    # graph on 16 vertices (Mantel's Theorem).  Triangle-free is necessary
    # because any triangle makes its edges non-essential (a length-2
    # alternative path exists through the third vertex).
    #
    # In K_{8,8}, removing any edge (a,b) with a in A, b in B disconnects
    # the graph (no length-2 path a-x-b exists since x would need to be
    # in both partitions simultaneously), so every edge is essential.
    half = 16 // 2
    edges = [[int(u), int(v)] for u in range(half) for v in range(half, 16)]
    return edges
# EVOLVE-BLOCK-END
