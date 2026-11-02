"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: i=0, A: [1, 1], B: [1, 1], W: [1/2, 0]
    U[0*2 + 0][0] = 1
    U[0*2 + 1][0] = 1
    for k in range(p):
        V[0*p + k][0] = 1
        V[1*p + k][0] = 1
        W[k*n + 0][0] = 0.5
        W[k*n + 1][0] = 0
        
    # r=1: i=0, A: [1, -1], B: [1, -1], W: [1/2, 0]
    U[0*2 + 0][1] = 1
    U[0*2 + 1][1] = -1
    for k in range(p):
        V[0*p + k][1] = 1
        V[1*p + k][1] = -1
        W[k*n + 0][1] = 0.5
        W[k*n + 1][1] = 0
        
    # r=2: i=1, A: [1, 1], B: [1, 1], W: [0, 1/2]
    U[1*2 + 0][2] = 1
    U[1*2 + 1][2] = 1
    for k in range(p):
        V[0*p + k][2] = 1
        V[1*p + k][2] = 1
        W[k*n + 0][2] = 0
        W[k*n + 1][2] = 0.5
        
    # r=3: i=1, A: [1, -1], B: [1, -1], W: [0, 1/2]
    U[1*2 + 0][3] = 1
    U[1*2 + 1][3] = -1
    for k in range(p):
        V[0*p + k][3] = 1
        V[1*p + k][3] = -1
        W[k*n + 0][3] = 0
        W[k*n + 1][3] = 0.5
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
