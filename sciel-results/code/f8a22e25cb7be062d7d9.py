"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Andrasfai graph And(9): Z_27 with i ~ j iff (j - i) mod 27 is 1 mod 3.
    # It is triangle-free and its largest independent set has 9 vertices.
    # n = 27, alpha = 9.
    n = 27
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
