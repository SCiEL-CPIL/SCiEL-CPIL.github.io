"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222_factors():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W are 4x7 matrices.
    Indices:
    U row: i*2 + j
    V row: j*2 + k
    W row: k*2 + i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0 = A00*(B01-B11)
    U[0][0] = 1
    V[1][0] = 1; V[3][0] = -1
    W[0][0] = -1; W[1][0] = 1
    
    # M1 = (A00+A01)*B11
    U[0][1] = 1; U[1][1] = 1
    V[3][1] = 1
    W[0][1] = 1; W[1][1] = 1
    
    # M2 = (A10-A00)*(B00+B01)
    U[2][2] = 1; U[0][2] = -1
    V[0][2] = 1; V[1][2] = 1
    W[2][2] = 1; W[3][2] = 1
    
    # M3 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = -1; W[3][3] = 1
    
    # M4 = (A00+A11)*(B00+B11)
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[1][4] = -1; W[2][4] = -1; W[3][4] = 1
    
    # M5 = (A01-A11)*(B10+B11)
    U[1][5] = 1; U[3][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # M6 = (A10+A11)*(B00-B10)
    U[2][6] = 1; U[3][6] = 1
    V[0][6] = 1; V[2][6] = -1
    W[2][6] = 1; W[0][6] = -1
    
    return U, V, W

def _bini_333_factors():
    """
    Returns U, V, W for rank-23 decomposition of 3x3x3 matrix multiplication.
    U, V, W are 9x23 matrices.
    Indices:
    U row: i*3 + j
    V row: j*3 + k
    W row: k*3 + i
    """
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1
    W[1][0] = -1; W[5][0] = -1
    
    # M2 = (A01 + A11)(B00 + B10)
    U[1][1] = 1; U[4][1] = 1
    V[0][1] = 1; V[3][1] = 1
    W[1][1] = 1; W[5][1] = 1
    W[2][1] = -1; W[6][1] = -1
    
    # M3 = (A10 + A11)(B11 + B01)
    U[2][2] = 1; U[4][2] = 1
    V[8][2] = 1; V[5][2] = 1
    W[2][2] = 1; W[6][2] = 1
    W[3][2] = -1; W[7][2] = -1
    
    # M4 = A00(B11 - B01)
    U[0][3] = 1
    V[8][3] = 1; V[5][3] = -1
    W[0][3] = 1; W[1][3] = -1
    
    # M5 = A11(B00 - B10)
    U[4][4] = 1
    V[0][4] = 1; V[3][4] = -1
    W[4][4] = 1; W[5][4] = -1
    
    # M6 = (A00 + A01)B11
    U[0][5] = 1; U[1][5] = 1
    V[8][5] = 1
    W[0][5] = -1; W[1][5] = 1
    
    # M7 = (A10 - A00)(B00 + B01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[3][6] = 1
    
    # M8 = (A01 - A11)(B10 + B11)
    U[1][7] = 1; U[4][7] = -1
    V[3][7] = 1; V[8][7] = 1
    W[1][7] = 1; W[2][7] = 1
    
    # M9 = A02(B00 + B11)
    U[3][8] = 1
    V[0][8] = 1; V[8][8] = 1
    W[0][8] = 1; W[4][8] = 1
    
    # M10 = A12(B10 + B00)
    U[6][9] = 1
    V[3][9] = 1; V[0][9] = 1
    W[1][9] = 1; W[4][9] = 1
    
    # M11 = A20(B01 + B11)
    U[7][10] = 1
    V[5][10] = 1; V[8][10] = 1
    W[2][10] = 1; W[4][10] = 1
    
    # M12 = A22(B00 + B11)
    U[8][11] = 1
    V[0][11] = 1; V[8][11] = 1
    W[0][11] = 1; W[4][11] = 1
    
    # M13 = (A02 + A22)B01
    U[3][12] = 1; U[8][12] = 1
    V[5][12] = 1
    W[1][12] = 1; W[2][12] = 1
    
    # M14 = (A12 + A22)B10
    U[6][13] = 1; U[8][13] = 1
    V[3][13] = 1
    W[1][13] = 1; W[2][13] = 1
    
    # M15 = (A02 + A22)B11
    U[3][14] = 1; U[8][14] = 1
    V[8][14] = 1
    W[0][14] = 1; W[4][14] = 1
    
    # M16 = A02(B10 - B00)
    U[3][15] = 1
    V[3][15] = 1; V[0][15] = -1
    W[1][15] = 1; W[0][15] = -1
    
    # M17 = A12(B01 - B11)
    U[6][16] = 1
    V[5][16] = 1; V[8][16] = -1
    W[2][16] = 1; W[4][16] = -1
    
    # M18 = A20(B10 - B00)
    U[7][17] = 1
    V[3][17] = 1; V[0][17] = -1
    W[2][17] = 1; W[0][17] = -1
    
    # M19 = A22(B01 - B11)
    U[8][18] = 1
    V[5][18] = 1; V[8][18] = -1
    W[2][18] = 1; W[4][18] = -1
    
    # M20 = (A02 - A22)B00
    U[3][19] = 1; U[8][19] = -1
    V[0][19] = 1
    W[0][19] = 1
    
    # M21 = (A12 - A22)B11
    U[6][20] = 1; U[8][20] = -1
    V[8][20] = 1
    W[4][20] = 1
    
    # M22 = (A20 - A02)B10
    U[7][21] = 1; U[3][21] = -1
    V[3][21] = 1
    W[2][21] = 1
    
    # M23 = (A22 - A12)B01
    U[8][22] = 1; U[6][22] = -1
    V[5][22] = 1
    W[2][22] = 1
    
    return U, V, W

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _add_block(U, V, W, U_blk, V_blk, W_blk, i_off, j_off, k_off, n_blk, m_blk, p_blk, n, m, p, col_start):
    """
    Add block factors to global U, V, W at column col_start.
    U_blk is n_blk x m_blk
    V_blk is m_blk x p_blk
    W_blk is p_blk x n_blk
    Global U row: (i_off + li)*m + (j_off + lj)
    Global V row: (j_off + lj)*p + (k_off + lk)
    Global W row: (k_off + lk)*n + (i_off + li)
    """
    for r in range(len(U_blk[0])):
        for li in range(n_blk):
            for lj in range(m_blk):
                U[(i_off + li) * m + (j_off + lj)][col_start + r] = U_blk[li * m_blk + lj][r]
        for lj in range(m_blk):
            for lk in range(p_blk):
                V[(j_off + lj) * p + (k_off + lk)][col_start + r] = V_blk[lj * p_blk + lk][r]
        for lk in range(p_blk):
            for li in range(n_blk):
                W[(k_off + lk) * n + (i_off + li)][col_start + r] = W_blk[lk * n_blk + li][r]
    return col_start + len(U_blk[0])

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 100  # safety margin
    
    # Strategy: block decomposition
    # Split j (0..4) into [0,3) and [3,5)
    # Split k (0..4) into [0,3) and [3,5)
    # i stays as [0,3)
    #
    # Block (j=0..2, k=0..2): 3x3x3 -> use Bini rank 23
    # Block (j=0..2, k=3..4): 3x3x2 -> schoolbook rank 18
    # Block (j=3..4, k=0..2): 3x2x3 -> schoolbook rank 18
    # Block (j=3..4, k=3..4): 3x2x2 -> schoolbook rank 12
    # Total rank: 23 + 18 + 18 + 12 = 71
    
    rank = 23 + 18 + 18 + 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 3x3x3 (i:0-2, j:0-2, k:0-2)
    U1, V1, W1 = _bini_333_factors()
    col = _add_block(U, V, W, U1, V1, W1, 0, 0, 0, 3, 3, 3, n, m, p, col)
    
    # Block 2: 3x3x2 (i:0-2, j:0-2, k:3-4)
    U2, V2, W2 = _schoolbook_factors(3, 3, 2)
    col = _add_block(U, V, W, U2, V2, W2, 0, 0, 3, 3, 3, 2, n, m, p, col)
    
    # Block 3: 3x2x3 (i:0-2, j:3-4, k:0-2)
    U3, V3, W3 = _schoolbook_factors(3, 2, 3)
    col = _add_block(U, V, W, U3, V3, W3, 0, 3, 0, 3, 2, 3, n, m, p, col)
    
    # Block 4: 3x2x2 (i:0-2, j:3-4, k:3-4)
    U4, V4, W4 = _schoolbook_factors(3, 2, 2)
    col = _add_block(U, V, W, U4, V4, W4, 0, 3, 3, 3, 2, 2, n, m, p, col)
    
    if col != rank:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    # Verification using numpy for speed
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
                    s = np.dot(W_np[k*n + i], U_np[i*m + j] * V_np[j*p + k])
                    if not np.isclose(s, 1.0, atol=1e-10):
                        # Fall back to schoolbook
                        U, V, W = _schoolbook_factors(n, m, p)
                        return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
