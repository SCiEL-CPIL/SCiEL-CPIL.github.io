"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    edges = [[0, v] for v in range(1, n)]
    return edges
# EVOLVE-BLOCK-END
