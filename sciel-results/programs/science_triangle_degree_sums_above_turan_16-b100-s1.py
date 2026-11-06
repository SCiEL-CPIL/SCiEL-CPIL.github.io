"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # K_{6,10} plus 5 disjoint edges in the size-10 part.
    # Part A: vertices 0-5 (size 6), Part B: vertices 6-15 (size 10)
    edges = [[u, v] for u in range(6) for v in range(6, 16)]
    # Add 5 disjoint edges in B: (6,7), (8,9), (10,11), (12,13), (14,15)
    edges += [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    return {'n': 16, 'edges': edges}
# EVOLVE-BLOCK-END
