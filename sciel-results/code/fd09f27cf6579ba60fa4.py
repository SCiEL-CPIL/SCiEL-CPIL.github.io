"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x4x7 multiplication into:
    # 1. 4x4x4 multiplication (columns 0-3 of B)
    # 2. 4x4x3 multiplication (columns 4-6 of B)
    
    # Rank for 4x4x4 is 16 (schoolbook)
    # Rank for 4x4x3 is 12 (schoolbook)
    # Total rank = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4
    # i, j, k in [0, 3]
    # U index: i*m + j
    # V index: j*p + k  (p=7 globally, k in 0..3)
    # W index: k*n + i
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(4): # First 4 columns
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
                
    # Part 2: 4x4x3
    # i, j in [0, 3], k in [0, 2]
    # Global column index for B and C is k + 4
    # U index: i*m + j
    # V index: j*p + (k+4)
    # W index: (k+4)*n + i
    for i in range(n):
        for j in range(m):
            for k in range(3): # Next 3 columns
                U[i * m + j][t] = 1
                V[j * p + (k + 4)][t] = 1
                W[(k + 4) * n + i][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
