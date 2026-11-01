"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i, j, k] = 1 for all i, j, k.
    # This tensor is the outer product of three all-ones vectors.
    # Thus, it has tensor rank 1.
    rank = 1
    U = [[1] * rank for _ in range(n)]
    V = [[1] * rank for _ in range(m)]
    W = [[1] * rank for _ in range(p)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
