"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: split A (4x4) into 2x2 blocks, B (4x9) into 2x9 blocks
    # C1 (2x9) = A11*B1 + A12*B2
    # C2 (2x9) = A21*B1 + A22*B2
    # Each 2x2 x 2x9 uses rank 36 (4*9 schoolbook)
    # Total rank = 4 * 36 = 144... no wait, 4 blocks * 36 = 144. Same as before.
    # Let me use 4 blocks of 2x2 x 2x9, each with rank 36. Total = 144.
    # That's the same. I need a better decomposition.
    
    # Actually, let me just do the full schoolbook but in a cleaner way.
    # The schoolbook rank is n*m*p = 4*4*9 = 144.
    
    rank = 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i,j,k), U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
