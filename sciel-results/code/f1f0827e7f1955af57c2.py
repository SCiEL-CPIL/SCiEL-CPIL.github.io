"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # r=0 (M1)
    U[0*2+0][0] = 1
    U[1*2+1][0] = 1
    V[0*2+0][0] = 1
    V[1*2+1][0] = 1
    W[0*2+0][0] = 1
    W[1*2+1][0] = 1
    
    # r=1 (M2)
    U[1*2+0][1] = 1
    U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1
    W[1*2+1][1] = -1
    
    # r=2 (M3)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1
    V[1*2+1][2] = -1
    W[1*2+0][2] = 1
    W[1*2+1][2] = 1
    
    # r=3 (M4)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1
    V[0*2+0][3] = -1
    W[0*2+0][3] = 1
    W[0*2+1][3] = 1
    
    # r=4 (M5)
    U[0*2+0][4] = 1
    U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1
    W[1*2+0][4] = 1
    
    # r=5 (M6)
    U[1*2+0][5] = 1
    U[0*2+0][5] = -1
    V[0*2+0][5] = 1
    V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # r=6 (M7)
    U[0*2+1][6] = 1
    U[1*2+1][6] = -1
    V[1*2+0][6] = 1
    V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
