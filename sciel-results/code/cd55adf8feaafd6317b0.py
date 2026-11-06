"""Optimized program: Returns the best known valid graph (n=32) immediately to ensure speed and reliability."""
import time

# EVOLVE-BLOCK-START
def construct():
    # The best known construction is 8 disjoint K4s.
    # n = 32
    # Edges: 6 per K4, total 48.
    # Properties:
    # - C5-free: K4 has no cycles of length > 3. Disjoint union has no cycles between components.
    # - Alpha: Max independent set in K4 is 1. Total alpha = 8 * 1 = 8 < 9.
    # This graph is valid and maximizes n among known constructions.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        # K4 on vertices base, base+1, base+2, base+3
        edges.append([base, base + 1])
        edges.append([base, base + 2])
        edges.append([base, base + 3])
        edges.append([base + 1, base + 2])
        edges.append([base + 1, base + 3])
        edges.append([base + 2, base + 3])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
