"""Rewritten program: rank-34 decomposition for 3x3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
    U: shape (4, 7)  indexed by (i*2+j)
    V: shape (4, 7)  indexed by (j*2+k)
    W: shape (4, 7)  indexed by (k*2+i)
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1

    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = -1; W[3][1] = 1

    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _direct_2x1x4():
    """
    Direct decomposition for 2x1 x 1x4 -> 2x4.
    A is 2x1 (column vector), B is 1x4 (row vector).
    C[i,k] = A[i,0] * B[0,k]
    Rank = 2*4 = 8.
    U: shape (2*1, 8) = (2, 8), indexed by i (since m=1, i*1+0 = i)
    V: shape (1*4, 8) = (4, 8), indexed by k (since m=1, 0*4+k = k)
    W: shape (4*2, 8) = (8, 8), indexed by (k*2+i)
    """
    rank = 8
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(8)]
    col = 0
    for i in range(2):
        for k in range(4):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    return U, V, W


def _direct_1x2x4():
    """
    Direct decomposition for 1x2 x 2x4 -> 1x4.
    A is 1x2 (row vector), B is 2x4.
    C[0,k] = sum_j A[0,j] * B[j,k]
    Rank = 1*2*4 = 8.
    U: shape (1*2, 8) = (2, 8), indexed by j (i=0, 0*2+j = j)
    V: shape (2*4, 8) = (8, 8), indexed by (j*4+k)
    W: shape (4*1, 8) = (4, 8), indexed by (k*1+0 = k)
    """
    rank = 8
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(4)]
    col = 0
    for j in range(2):
        for k in range(4):
            U[j][col] = 1
            V[j*4 + k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W


def _direct_1x1x4():
    """
    Direct decomposition for 1x1 x 1x4 -> 1x4.
    Rank = 4.
    U: shape (1, 4)
    V: shape (4, 4)
    W: shape (4, 4)
    """
    rank = 4
    U = [[0]*rank for _ in range(1)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    for k in range(4):
        U[0][k] = 1
        V[k][k] = 1
        W[k][k] = 1
    return U, V, W


def _embed_2x2x2_in_3x3x4(U2, V2, W2, i_off, j_off, k_off):
    """
    Embed a 2x2x2 Strassen decomposition into the 3x3x4 tensor.
    U2: (4, 7) for A[i_off:i_off+2, j_off:j_off+2]
    V2: (4, 7) for B[j_off:j_off+2, k_off:k_off+2]
    W2: (4, 7) for C[k_off+k, i_off+i]
    
    Returns U, V, W for the full 3x3x4 problem:
    U: (9, 7), V: (12, 7), W: (12, 7)
    """
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(12)]
    W = [[0]*7 for _ in range(12)]
    
    for li in range(2):
        for lj in range(2):
            idx_u = (i_off + li) * 3 + (j_off + lj)
            for c in range(7):
                U[idx_u][c] = U2[li*2 + lj][c]
    
    for lj in range(2):
        for lk in range(2):
            idx_v = (j_off + lj) * 4 + (k_off + lk)
            for c in range(7):
                V[idx_v][c] = V2[lj*2 + lk][c]
    
    for lk in range(2):
        for li in range(2):
            idx_w = (k_off + lk) * 3 + (i_off + li)
            for c in range(7):
                W[idx_w][c] = W2[lk*2 + li][c]
    
    return U, V, W


def _embed_2x1x4_in_3x3x4(U2, V2, W2, i_off, j_fixed, k_off):
    """
    Embed a 2x1x4 direct decomposition into the 3x3x4 tensor.
    A[i_off:i_off+2, j_fixed], B[j_fixed, k_off:k_off+4], C[k_off+k, i_off+i]
    U2: (2, 8), V2: (4, 8), W2: (8, 8)
    """
    U = [[0]*8 for _ in range(9)]
    V = [[0]*8 for _ in range(12)]
    W = [[0]*8 for _ in range(12)]
    
    for li in range(2):
        idx_u = (i_off + li) * 3 + j_fixed
        for c in range(8):
            U[idx_u][c] = U2[li][c]
    
    for lk in range(4):
        idx_v = j_fixed * 4 + (k_off + lk)
        for c in range(8):
            V[idx_v][c] = V2[lk][c]
    
    for lk in range(4):
        for li in range(2):
            idx_w = (k_off + lk) * 3 + (i_off + li)
            for c in range(8):
                W[idx_w][c] = W2[lk*2 + li][c]
    
    return U, V, W


def _embed_1x2x4_in_3x3x4(U2, V2, W2, i_fixed, j_off, k_off):
    """
    Embed a 1x2x4 direct decomposition into the 3x3x4 tensor.
    A[i_fixed, j_off:j_off+2], B[j_off:j_off+2, k_off:k_off+4], C[k_off+k, i_fixed]
    U2: (2, 8), V2: (8, 8), W2: (4, 8)
    """
    U = [[0]*8 for _ in range(9)]
    V = [[0]*8 for _ in range(12)]
    W = [[0]*8 for _ in range(12)]
    
    for lj in range(2):
        idx_u = i_fixed * 3 + (j_off + lj)
        for c in range(8):
            U[idx_u][c] = U2[lj][c]
    
    for lj in range(2):
        for lk in range(4):
            idx_v = (j_off + lj) * 4 + (k_off + lk)
            for c in range(8):
                V[idx_v][c] = V2[lj*4 + lk][c]
    
    for lk in range(4):
        idx_w = (k_off + lk) * 3 + i_fixed
        for c in range(8):
            W[idx_w][c] = W2[lk][c]
    
    return U, V, W


def _embed_1x1x4_in_3x3x4(U2, V2, W2, i_fixed, j_fixed, k_off):
    """
    Embed a 1x1x4 direct decomposition into the 3x3x4 tensor.
    A[i_fixed, j_fixed], B[j_fixed, k_off:k_off+4], C[k_off+k, i_fixed]
    U2: (1, 4), V2: (4, 4), W2: (4, 4)
    """
    U = [[0]*4 for _ in range(9)]
    V = [[0]*4 for _ in range(12)]
    W = [[0]*4 for _ in range(12)]
    
    idx_u = i_fixed * 3 + j_fixed
    for c in range(4):
        U[idx_u][c] = U2[0][c]
    
    for lk in range(4):
        idx_v = j_fixed * 4 + (k_off + lk)
        for c in range(4):
            V[idx_v][c] = V2[lk][c]
    
    for lk in range(4):
        idx_w = (k_off + lk) * 3 + i_fixed
        for c in range(4):
            W[idx_w][c] = W2[lk][c]
    
    return U, V, W


def _add_matrices(A, B, rank):
    """Add two matrices of shape (rows, rank) element-wise."""
    rows = len(A)
    C = [[0]*rank for _ in range(rows)]
    for r in range(rows):
        for c in range(rank):
            C[r][c] = A[r][c] + B[r][c]
    return C


def construct():
    """
    Rank-34 decomposition for 3x3 x 3x4 matrix multiplication.
    
    Breakdown:
    1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : rank 7 (Strassen 2x2x2)
    2. A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] : rank 7 (Strassen 2x2x2)
    3. A[0:2, 2] * B[2, :] -> C[:, 0:2] : rank 8 (direct 2x1x4)
    4. A[2, 0:2] * B[0:2, :] -> C[:, 2] : rank 8 (direct 1x2x4)
    5. A[2, 2] * B[2, :] -> C[:, 3] : rank 4 (direct 1x1x4)
    
    Total rank: 7 + 7 + 8 + 8 + 4 = 34
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    U_full = [[0]*34 for _ in range(n*m)]   # (9, 34)
    V_full = [[0]*34 for _ in range(m*p)]   # (12, 34)
    W_full = [[0]*34 for _ in range(p*n)]   # (12, 34)
    
    col_offset = 0
    
    # Component 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i_off=0, j_off=0, k_off=0
    U1, V1, W1 = _strassen_2x2x2()
    U1e, V1e, W1e = _embed_2x2x2_in_3x3x4(U1, V1, W1, 0, 0, 0)
    for r in range(9):
        for c in range(7):
            U_full[r][col_offset + c] += U1e[r][c]
    for r in range(12):
        for c in range(7):
            V_full[r][col_offset + c] += V1e[r][c]
    for r in range(12):
        for c in range(7):
            W_full[r][col_offset + c] += W1e[r][c]
    col_offset += 7
    
    # Component 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    # i_off=0, j_off=0, k_off=2
    U2, V2, W2 = _strassen_2x2x2()
    U2e, V2e, W2e = _embed_2x2x2_in_3x3x4(U2, V2, W2, 0, 0, 2)
    for r in range(9):
        for c in range(7):
            U_full[r][col_offset + c] += U2e[r][c]
    for r in range(12):
        for c in range(7):
            V_full[r][col_offset + c] += V2e[r][c]
    for r in range(12):
        for c in range(7):
            W_full[r][col_offset + c] += W2e[r][c]
    col_offset += 7
    
    # Component 3: A[0:2, 2] * B[2, :] -> C[:, 0:2]
    # i_off=0, j_fixed=2, k_off=0
    U3, V3, W3 = _direct_2x1x4()
    U3e, V3e, W3e = _embed_2x1x4_in_3x3x4(U3, V3, W3, 0, 2, 0)
    for r in range(9):
        for c in range(8):
            U_full[r][col_offset + c] += U3e[r][c]
    for r in range(12):
        for c in range(8):
            V_full[r][col_offset + c] += V3e[r][c]
    for r in range(12):
        for c in range(8):
            W_full[r][col_offset + c] += W3e[r][c]
    col_offset += 8
    
    # Component 4: A[2, 0:2] * B[0:2, :] -> C[:, 2]
    # i_fixed=2, j_off=0, k_off=0
    # Wait, this needs to produce C[k, 2] for all k=0..3
    # C[k, 2] = sum_j A[2, j] * B[j, k] for j=0,1
    # This is 1x2 x 2x4 -> 1x4, giving C[k, 2]
    U4, V4, W4 = _direct_1x2x4()
    U4e, V4e, W4e = _embed_1x2x4_in_3x3x4(U4, V4, W4, 2, 0, 0)
    for r in range(9):
        for c in range(8):
            U_full[r][col_offset + c] += U4e[r][c]
    for r in range(12):
        for c in range(8):
            V_full[r][col_offset + c] += V4e[r][c]
    for r in range(12):
        for c in range(8):
            W_full[r][col_offset + c] += W4e[r][c]
    col_offset += 8
    
    # Component 5: A[2, 2] * B[2, :] -> C[:, 3]
    # i_fixed=2, j_fixed=2, k_off=0
    U5, V5, W5 = _direct_1x1x4()
    U5e, V5e, W5e = _embed_1x1x4_in_3x3x4(U5, V5, W5, 2, 2, 0)
    for r in range(9):
        for c in range(4):
            U_full[r][col_offset + c] += U5e[r][c]
    for r in range(12):
        for c in range(4):
            V_full[r][col_offset + c] += V5e[r][c]
    for r in range(12):
        for c in range(4):
            W_full[r][col_offset + c] += W5e[r][c]
    col_offset += 4
    
    assert col_offset == 34, f"Expected rank 34, got {col_offset}"
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
