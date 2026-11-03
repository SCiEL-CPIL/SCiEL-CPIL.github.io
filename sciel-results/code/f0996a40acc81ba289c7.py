"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0 corresponds to i=0
    for j in range(m):
        U[0 * m + j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    # r=1 corresponds to i=1
    for j in range(m):
        U[1 * m + j][1] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][1] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
