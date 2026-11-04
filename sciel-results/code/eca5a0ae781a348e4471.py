"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # The 2x4x6 matrix multiplication structure tensor is the all-ones tensor.
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is a rank-1 tensor: T = U ⊗ V ⊗ W with all entries 1.
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
