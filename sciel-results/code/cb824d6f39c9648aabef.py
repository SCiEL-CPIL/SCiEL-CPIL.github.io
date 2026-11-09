"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 29 using Strassen for 2x2x2 sub-block
    n, m, p = DIMENSIONS
    rank = 29
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Part 1: A[:, 0] * B[0, :] (2x1 * 1x5) -> Rank 10
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col_idx] = 1.0
            V[0 * p + k][col_idx] = 1.0
            W[k * n + i][col_idx] = 1.0
            col_idx += 1
            
    # Part 2: A[:, 1:3] * B[1:3, 0:2] (2x2 * 2x2) -> Rank 7 (Strassen)
    # U indices: A[0,1]=1, A[0,2]=2, A[1,1]=4, A[1,2]=5
    # V indices: B[1,0]=5, B[1,1]=6, B[2,0]=10, B[2,1]=11
    # W indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    # M1 = (A01 + A12) * (B10 + B21)
    U[1][col_idx] = 1.0; U[5][col_idx] = 1.0
    V[5][col_idx] = 1.0; V[11][col_idx] = 1.0
    W[0][col_idx] = 1.0; W[3][col_idx] = 1.0
    col_idx += 1
    
    # M2 = (A11 + A12) * B10
    U[4][col_idx] = 1.0; U[5][col_idx] = 1.0
    V[5][col_idx] = 1.0
    W[1][col_idx] = 1.0; W[3][col_idx] = -1.0
    col_idx += 1
    
    # M3 = A01 * (B11 - B21)
    U[1][col_idx] = 1.0
    V[6][col_idx] = 1.0; V[11][col_idx] = -1.0
    W[2][col_idx] = 1.0; W[3][col_idx] = 1.0
    col_idx += 1
    
    # M4 = A12 * (B20 - B10)
    U[5][col_idx] = 1.0
    V[10][col_idx] = 1.0; V[5][col_idx] = -1.0
    W[0][col_idx] = 1.0; W[1][col_idx] = 1.0
    col_idx += 1
    
    # M5 = (A01 + A11) * B21
    U[1][col_idx] = 1.0; U[4][col_idx] = 1.0
    V[11][col_idx] = 1.0
    W[0][col_idx] = -1.0; W[2][col_idx] = 1.0
    col_idx += 1
    
    # M6 = (A02 - A01) * (B10 + B11)
    U[2][col_idx] = 1.0; U[1][col_idx] = -1.0
    V[5][col_idx] = 1.0; V[6][col_idx] = 1.0
    W[3][col_idx] = 1.0
    col_idx += 1
    
    # M7 = (A12 - A11) * (B20 + B21)
    U[5][col_idx] = 1.0; U[4][col_idx] = -1.0
    V[10][col_idx] = 1.0; V[11][col_idx] = 1.0
    W[0][col_idx] = 1.0
    col_idx += 1
    
    # Part 3: A[:, 1:3] * B[1:3, 2:5] (2x2 * 2x3) -> Rank 12 (Schoolbook)
    for i in range(n):
        for j_offset in range(2):
            j_actual = 1 + j_offset
            for k in range(2, p):
                U[i * m + j_actual][col_idx] = 1.0
                V[j_actual * p + k][col_idx] = 1.0
                W[k * n + i][col_idx] = 1.0
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
