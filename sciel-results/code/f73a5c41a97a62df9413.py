"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # 10 for p=5
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 10)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 10)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 10)
    
    # For i=0: terms c=0..p-1
    for c in range(p):
        for j in range(m):
            U[0 * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                if k == c:
                    V[j * p + k][c] = 1
        for k in range(p):
            W[k * n + 0][c] = 1
    
    # For i=1: terms c=p..2p-1
    for c in range(p, 2 * p):
        c_local = c - p
        for j in range(m):
            U[1 * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                if k == c_local:
                    V[j * p + k][c] = 1
        for k in range(p):
            W[k * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
