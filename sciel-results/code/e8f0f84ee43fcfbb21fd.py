"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition
    rank = m * p  # 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r corresponds to (j, k)
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U factors
            # L_A = A[0,j] + (k+1)A[1,j]
            U[0 * m + j][r] = 1
            U[1 * m + j][r] = k + 1
            
            # V factors
            # L_B = B[j,k]
            V[j * p + k][r] = 1
            
            # W factors
            # We derived that only k'=0 and k'=1 contribute to W
            # W[2k + i, (j, k')]
            # For i=0 (row 0 of C):
            # W[2k, (j,0)] = 2
            # W[2k, (j,1)] = -1
            # For i=1 (row 1 of C):
            # W[2k+1, (j,0)] = -1
            # W[2k+1, (j,1)] = 1
            
            # k' = 0
            r0 = j * p + 0
            W[2 * k + 0][r0] = 2
            W[2 * k + 1][r0] = -1
            
            # k' = 1
            r1 = j * p + 1
            W[2 * k + 0][r1] = -1
            W[2 * k + 1][r1] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
