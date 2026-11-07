"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Constructs a diameter-2 graph on n=16 vertices with maximum edges.
    Uses a complete bipartite graph K_{8,8}.
    This graph is triangle-free, ensuring all edges are essential for diameter 2.
    Total edges: 8 * 8 = 64.
    """
    n = P['n']
    if n < 2:
        return []
    
    # Split vertices into two equal parts
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    for u in part1:
        for v in part2:
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
