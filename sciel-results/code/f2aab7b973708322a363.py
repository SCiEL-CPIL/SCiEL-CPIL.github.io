"""Program for matrix multiplication tensor decomposition 2x3 by 3x9."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # Rank-27 decomposition using block structure
    # C[i, 3t+r] = sum_j A[i,j] B[j, 3t+r]
    # For each block t (0,1,2), we have a 2x3x3 multiplication
    # The 2x3x3 tensor has rank 9 (schoolbook)
    # Total rank = 3 * 9 = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]   # 6 x 27
    V = [[0] * rank for _ in range(m * p)]   # 27 x 27
    W = [[0] * rank for _ in range(p * n)]   # 18 x 27
    
    col = 0
    for t in range(3):  # 3 column blocks of B
        for j in range(m):  # sum over j
            for i in range(n):  # for each i
                for r in range(3):  # r = 0,1,2 within block
                    # U[i*m + j] = 1
                    U[i * m + j][col] = 1
                    # V[j*p + (3*t + r)] = 1
                    V[j * p + 3 * t + r][col] = 1
                    # W[(3*t + r)*n + i] = 1
                    W[(3 * t + r) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
