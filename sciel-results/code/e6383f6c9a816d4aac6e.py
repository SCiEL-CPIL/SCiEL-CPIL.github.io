"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Laderman's algorithm for 2x3x3 (rank 7) for each of the 3 blocks.
    # Total rank = 3 * 7 = 21.
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman factors for 2x3x3 matrix multiplication
    # U_L is 6x7, V_L is 9x7, W_L is 6x7
    # Indices: U_L[i*3+j, r], V_L[j*3+k, r], W_L[k*2+i, r]
    
    # Hardcoded Laderman factors for rank 7
    # This is a known decomposition.
    U_L = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]
    ]
    V_L = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    W_L = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]
    ]
    
    # Note: The above hardcoded factors are not correct for Laderman.
    # I will use a valid rank-7 decomposition if I can find it.
    # Otherwise, I will use a rank-9 decomposition for 2x3x3.
    # For now, I will use the schoolbook for 2x3x3 which is rank 18.
    # Total rank = 3 * 18 = 54. This is not an improvement.
    # I need a better decomposition.
    
    # Let's use a rank-9 decomposition for 2x3x3 if possible.
    # Actually, the rank of 2x3x3 is 7.
    # I will try to find the correct Laderman factors.
    
    # For now, I will return the schoolbook decomposition of rank 54.
    # This is safe but not an improvement.
    # I will try to implement a rank-27 decomposition using rank-9 for 2x3x3.
    
    # Rank 9 for 2x3x3:
    # This is possible by grouping terms.
    # Let's use a known rank-9 decomposition.
    
    # Actually, I will use the schoolbook for 2x3x3 (rank 18) and see if I can reduce it.
    # For now, I will return the schoolbook decomposition of rank 54.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
