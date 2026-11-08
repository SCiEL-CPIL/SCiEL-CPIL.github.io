"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Block 1: j in {0,1}, use 2 terms per k → 2*7 = 14 terms
    # Block 2: j = 2, use 1 term per k → 7 terms
    # Total rank = 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 6 x 21
    V = [[0] * rank for _ in range(m * p)]  # 21 x 21
    W = [[0] * rank for _ in range(p * n)]  # 14 x 21
    
    # Block 1: for each k in 0..6, for each s in {0,1}
    for k in range(p):
        for s in range(n):  # s in {0,1}
            r = 2 * k + s
            # U[i*m + j, r] = 1 if i == s and j in {0,1}
            for i in range(n):
                if i == s:
                    for j in range(2):  # j in {0,1}
                        U[i * m + j][r] = 1
            # V[j*p + k, r] = 1 for j in {0,1}
            for j in range(2):
                V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Block 2: for each k in 0..6
    for k in range(p):
        r = 14 + k
        # U[i*m + 2, r] = 1 for all i
        for i in range(n):
            U[i * m + 2][r] = 1
        # V[2*p + k, r] = 1
        V[2 * p + k][r] = 1
        # W[k*n + i, r] = 1 for all i
        for i in range(n):
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
