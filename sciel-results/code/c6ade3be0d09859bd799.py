"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Block schoolbook: split A (2x4) into two 2x2 blocks, B (4x5) into two 2x5 blocks
    # C = A1*B1 + A2*B2, each 2x2x5 done by schoolbook (rank 10 each)
    # Total rank: 20
    
    rank = 20
    U = [[0]*rank for _ in range(n*m)]   # 8 x 20
    V = [[0]*rank for _ in range(m*p)]   # 20 x 20
    W = [[0]*rank for _ in range(p*n)]   # 10 x 20
    
    col = 0
    # Block 1: A cols 0-1, B rows 0-1
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    
    # Block 2: A cols 2-3, B rows 2-3
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i*4 + 2 + j][col] = 1
                V[(2 + j)*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
