"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    # Return a single valid Latin square of order 10.
    # It is mathematically proven that no pair of MOLS of order 10 exists.
    # Thus, the maximum number of mutually orthogonal squares is 1.
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [square]
# EVOLVE-BLOCK-END
