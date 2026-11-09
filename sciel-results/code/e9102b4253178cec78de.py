"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Factor the all-ones 2x3x6 tensor as (2x3x1) kron (1x3x6).
    # rank(2x3x1 all-ones) = 2, rank(1x3x6 all-ones) = 3, so rank <= 6.
    n, m, p = DIMENSIONS
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for j in range(m):       # j = 0,1,2
        for c in range(2):   # c = 0,1
            # U: constant in i (A is constant in first index)
            for i in range(n):
                U[i * m + j][col] = 1
            # V: constant in k (B is constant in third index)
            for k in range(p):
                V[j * p + k][col] = 1
            # W: encodes (j, c) structure
            for k in range(p):
                for i in range(n):
                    W[k * n + i][col] = 1 if (i == c) else 0
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
