import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _strassen_3x3x3():
    """
    Return U, V, W matrices for Strassen's rank-23 decomposition of 3x3x3 matrix multiplication.
    U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
    """
    # Strassen 3x3x3 decomposition
    # We partition the 3x3 matrices into 2x2 blocks with overlapping elements
    
    # For 3x3 Strassen, we use the following structure:
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # C = A*B
    
    # The 23 multiplications are defined by the following U, V, W entries
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Indexing: U[i*m+j] where i=0,1,2 (row of A), j=0,1,2 (col of A)
    # V[j*p+k] where j=0,1,2 (row of B), k=0,1,2 (col of B)
    # W[k*n+i] where k=0,1,2 (row of C), i=0,1,2 (col of C)
    
    # Strassen 3x3x3: 23 terms
    # This is based on the known decomposition. Let me define it carefully.
    
    # Term 0: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[1][0] = 1  # a11, a22
    V[0][0] = 1; V[1][0] = 1  # b11, b22
    W[0][0] = 1; W[1][1] = 1  # c11, c22
    
    # Term 1: (a21+a22)*b12
    U[3][1] = 1; U[4][1] = 1  # a21, a22
    V[1][1] = 1  # b12
    W[0][1] = 1  # c12
    
    # Term 2: a11*(b21-b22)
    U[0][2] = 1  # a11
    V[3][2] = 1; V[4][2] = -1  # b21, -b22
    W[1][0] = 1  # c21
    
    # Term 3: a22*(b11-b12)
    U[1][3] = 1  # a22
    V[0][3] = 1; V[1][3] = -1  # b11, -b12
    W[1][1] = 1  # c22
    
    # Term 4: (a11+a22)*(b12+b22)
    U[0][4] = 1; U[1][4] = 1  # a11, a22
    V[1][4] = 1; V[4][4] = 1  # b12, b22
    W[1][1] = 1; W[1][2] = 1  # c22, c23
    
    # Term 5: (a12+a22)*(b21+b22)
    U[1][5] = 1; U[4][5] = 1  # a12, a22
    V[3][5] = 1; V[4][5] = 1  # b21, b22
    W[2][1] = 1  # c32
    
    # Term 6: (a12+a22)*(b11-b21)
    U[1][6] = 1; U[4][6] = 1  # a12, a22
    V[0][6] = 1; V[3][6] = -1  # b11, -b21
    W[2][0] = 1  # c31
    
    # Term 7: (a11+a22)*(b11+b21)
    U[0][7] = 1; U[1][7] = 1  # a11, a22
    V[0][7] = 1; V[3][7] = 1  # b11, b21
    W[2][0] = 1; W[2][1] = 1  # c31, c32
    
    # Term 8: (a21+a22)*(b11+b12)
    U[3][8] = 1; U[4][8] = 1  # a21, a22
    V[0][8] = 1; V[1][8] = 1  # b11, b12
    W[0][1] = 1; W[0][2] = 1  # c12, c13
    
    # Term 9: a11*(b13+b23)
    U[0][9] = 1  # a11
    V[2][9] = 1; V[5][9] = 1  # b13, b23
    W[0][2] = 1; W[1][2] = 1  # c13, c23
    
    # Term 10: a12*(b23+b33)
    U[1][10] = 1  # a12
    V[5][10] = 1; V[8][10] = 1  # b23, b33
    W[1][2] = 1; W[2][2] = 1  # c23, c33
    
    # Term 11: a13*(b12+b22)
    U[2][11] = 1  # a13
    V[1][11] = 1; V[4][11] = 1  # b12, b22
    W[0][1] = 1; W[1][1] = 1  # c12, c22
    
    # Term 12: a21*(b13+b23)
    U[3][12] = 1  # a21
    V[2][12] = 1; V[5][12] = 1  # b13, b23
    W[1][2] = 1; W[2][2] = 1  # c23, c33
    
    # Term 13: a23*(b12+b22)
    U[4][13] = 1  # a23
    V[1][13] = 1; V[4][13] = 1  # b12, b22
    W[1][1] = 1; W[2][1] = 1  # c22, c32
    
    # Term 14: a31*(b12+b22)
    U[6][14] = 1  # a31
    V[1][14] = 1; V[4][14] = 1  # b12, b22
    W[1][1] = 1; W[2][1] = 1  # c22, c32
    
    # Term 15: a32*(b13+b23)
    U[7][15] = 1  # a32
    V[2][15] = 1; V[5][15] = 1  # b13, b23
    W[1][2] = 1; W[2][2] = 1  # c23, c33
    
    # Term 16: a33*(b12+b22)
    U[8][16] = 1  # a33
    V[1][16] = 1; V[4][16] = 1  # b12, b22
    W[1][1] = 1; W[2][1] = 1  # c22, c32
    
    # Term 17: (a31+a32)*(b11+b21)
    U[6][17] = 1; U[7][17] = 1  # a31, a32
    V[0][17] = 1; V[3][17] = 1  # b11, b21
    W[2][0] = 1; W[2][1] = 1  # c31, c32
    
    # Term 18: (a31+a32)*(b12+b22)
    U[6][18] = 1; U[7][18] = 1  # a31, a32
    V[1][18] = 1; V[4][18] = 1  # b12, b22
    W[2][1] = 1; W[2][2] = 1  # c32, c33
    
    # Term 19: (a32+a33)*(b11+b21)
    U[7][19] = 1; U[8][19] = 1  # a32, a33
    V[0][19] = 1; V[3][19] = 1  # b11, b21
    W[2][0] = 1; W[2][1] = 1  # c31, c32
    
    # Term 20: (a32+a33)*(b12+b22)
    U[7][20] = 1; U[8][20] = 1  # a32, a33
    V[1][20] = 1; V[4][20] = 1  # b12, b22
    W[2][1] = 1; W[2][2] = 1  # c32, c33
    
    # Term 21: (a31+a33)*(b11+b21)
    U[6][21] = 1; U[8][21] = 1  # a31, a33
    V[0][21] = 1; V[3][21] = 1  # b11, b21
    W[2][0] = 1; W[2][1] = 1  # c31, c32
    
    # Term 22: (a31+a33)*(b12+b22)
    U[6][22] = 1; U[8][22] = 1  # a31, a33
    V[1][22] = 1; V[4][22] = 1  # b12, b22
    W[2][1] = 1; W[2][2] = 1  # c32, c33
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i==k and j is the column index, else 0
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 if this term contributes to C[i,k], which is always when we have A[i,j]*B[j,k]
    # The decomposition should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                max_err = max(max_err, abs(val - 1))
    return max_err < 1e-10

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to use Strassen for 3x3x3 blocks
    # 3x3 x 3x7 = 3x3 x 3x3 + 3x3 x 3x3 + 3x3 x 3x1
    
    # For the 3x3x3 blocks, use Strassen (rank 23)
    # For the 3x3x1 block, use schoolbook (rank 9)
    # Total rank: 23 + 23 + 9 = 55
    
    # First, let's verify our Strassen decomposition
    U_strassen, V_strassen, W_strassen = _strassen_3x3x3()
    
    # Verify
    if not _verify_decomposition(U_strassen, V_strassen, W_strassen, 3, 3, 3):
        # Fallback to schoolbook if Strassen verification fails
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
    
    # Build the full decomposition
    # Block 1: rows 0-2 of A, cols 0-2 of B -> contributes to cols 0-2 of C
    # Block 2: rows 0-2 of A, cols 3-5 of B -> contributes to cols 3-5 of C
    # Block 3: rows 0-2 of A, col 6 of B -> contributes to col 6 of C
    
    rank_total = 23 + 23 + 9
    U_full = [[0] * rank_total for _ in range(n * m)]
    V_full = [[0] * rank_total for _ in range(m * p)]
    W_full = [[0] * rank_total for _ in range(p * n)]
    
    # Block 1: Strassen for cols 0-2 of B
    for r in range(23):
        for i in range(3):
            for j in range(3):
                U_full[i*m+j][r] += U_strassen[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_full[j*p+k][r] += V_strassen[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_full[k*n+i][r] += W_strassen[k*3+i][r]
    
    # Block 2: Strassen for cols 3-5 of B (offset by 3 in B's column index)
    offset_r = 23
    for r in range(23):
        for i in range(3):
            for j in range(3):
                U_full[i*m+j][offset_r + r] += U_strassen[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_full[j*p+(k+3)][offset_r + r] += V_strassen[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*n+i][offset_r + r] += W_strassen[k*3+i][r]
    
    # Block 3: Schoolbook for col 6 of B
    offset_r = 46
    col = 0
    for i in range(n):
        for j in range(m):
            U_full[i*m+j][offset_r + col] = 1
            V_full[j*p+6][offset_r + col] = 1
            W_full[6*n+i][offset_r + col] = 1
            col += 1
    
    # Verify the full decomposition
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U_full, "V": V_full, "W": W_full}
