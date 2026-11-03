import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6, atol=1e-9, seed=42):
    """Verify that the decomposition is correct by checking on random matrices."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)

def _build_rank22_3x3x3():
    """
    Build a rank-22 decomposition for 3x3x3 matrix multiplication.
    U[i*3+j], V[j*3+k], W[k*3+i]
    """
    r = 22
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms (indices 0-6)
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Schoolbook 15 terms (indices 7-21)
    # C02 = A00 B02 + A01 B12 + A02 B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1

    # C12 = A10 B02 + A11 B12 + A12 B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1

    # C20 = A20 B00 + A21 B10 + A22 B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1

    # C21 = A20 B01 + A21 B11 + A22 B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1

    # C22 = A20 B02 + A21 B12 + A22 B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 3, 3, 6
    
    # Build two rank-22 decompositions for 3x3x3
    U1, V1, W1 = _build_rank22_3x3x3()
    U2, V2, W2 = _build_rank22_3x3x3()
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy U1 and U2 into U
    # U is 9x44. U1 is 9x22.
    for i in range(9):
        for r in range(22):
            U[i][r] = U1[i][r]
            U[i][r + 22] = U2[i][r]
            
    # Copy V1 and V2 into V
    # V is 18x44. V1 is 9x22 (for B cols 0-2). V2 is 9x22 (for B cols 3-5).
    # V index in 3x6 is j*6 + k.
    # V1 corresponds to k=0,1,2. So V1[j*3+k] maps to V[j*6+k].
    # V2 corresponds to k=0,1,2 in its local frame, but maps to k=3,4,5 in global.
    # So V2[j*3+k] maps to V[j*6 + (k+3)].
    
    for j in range(3):
        for k in range(3):
            local_idx = j * 3 + k
            global_idx_1 = j * 6 + k
            global_idx_2 = j * 6 + (k + 3)
            
            for r in range(22):
                V[global_idx_1][r] = V1[local_idx][r]
                V[global_idx_2][r + 22] = V2[local_idx][r]
                
    # Copy W1 and W2 into W
    # W is 18x44. W1 is 9x22 (for C cols 0-2). W2 is 9x22 (for C cols 3-5).
    # W index in 3x6 is k*3 + i.
    # W1 corresponds to k=0,1,2. So W1[k*3+i] maps to W[k*3+i].
    # W2 corresponds to k=0,1,2 in its local frame, but maps to k=3,4,5 in global.
    # So W2[k*3+i] maps to W[(k+3)*3 + i].
    
    for k in range(3):
        for i in range(3):
            local_idx = k * 3 + i
            global_idx_1 = k * 3 + i
            global_idx_2 = (k + 3) * 3 + i
            
            for r in range(22):
                W[global_idx_1][r] = W1[local_idx][r]
                W[global_idx_2][r + 22] = W2[local_idx][r]
                
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
