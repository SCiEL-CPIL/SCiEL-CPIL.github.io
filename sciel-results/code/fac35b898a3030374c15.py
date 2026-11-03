"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x6 multiplication into two 3x3 multiplications.
    # Rank will be 9 + 9 = 18.
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First 3x3 block (columns 0-2 of B and C)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Second 3x3 block (columns 3-5 of B and C)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U index: i*m + j
                # V index: j*p + (k+3)
                # W index: (k+3)*n + i
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
