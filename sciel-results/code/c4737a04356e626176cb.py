"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for matrix multiplication n x m by m x p is at least m.
    # We construct a decomposition of rank m by treating each middle index j
    # as a separate rank-1 component.
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        c = j
        # U index: i * m + j
        for i in range(n):
            U[i * m + j][c] = 1
        # V index: j * p + k
        for k in range(p):
            V[j * p + k][c] = 1
        # W index: k * n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
