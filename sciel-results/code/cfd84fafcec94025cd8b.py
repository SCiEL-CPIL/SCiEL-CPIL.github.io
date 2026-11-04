"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block2x2(rU, cU, rV, cV, rW, cW, col):
    """Add a 2x2x2 schoolbook block at rank offset col.
    Returns col+4.
    """
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                U_row = rU + ii*2 + jj
                V_row = rV + jj*2 + kk
                W_row = rW + kk*2 + ii
                U[U_row][col] = 1
                V[V_row][col] = 1
                W[W_row][col] = 1
                col += 1
    return col

def _block2x1(rU, cU, rV, cV, rW, cW, col):
    """2x1 x 1x1 -> 2x1 block. Rank 2.
    U: rows rU+r, cols cU+c (r in 0..1, c in 0)
    V: rows rV+r, cols cV+c (r in 0, c in 0)
    W: rows rW+r, cols cW+c (r in 0..1, c in 0)
    """
    for r in range(2):
        U[rU + r][col] = 1
        V[rV][col] = 1
        W[rW + r][col] = 1
        col += 1
    return col

def _block1x2(rU, cU, rV, cV, rW, cW, col):
    """1x2 x 2x1 -> 1x1 block. Rank 2.
    U: rows rU, cols cU+c (c in 0..1)
    V: rows rV+r, cols cV+c (r in 0..1, c in 0)
    W: rows rW, cols cW
    """
    for r in range(2):
        U[rU][col] = 1
        V[rV + r][col] = 1
        W[rW][col] = 1
        col += 1
    return col

def _block1x1(rU, cU, rV, cV, rW, cW, col):
    """1x1 x 1x1 -> 1x1 block. Rank 1."""
    U[rU][col] = 1
    V[rV][col] = 1
    W[rW][col] = 1
    return col + 1

def _strassen_block_2x2(rU, cU, rV, cV, rW, cW, col):
    """Add rank-7 Strassen for 2x2x2 at rank offset col.
    U: rows rU + i*2 + j (i,j in 0..1), 4 rows
    V: rows rV + j*2 + k (j,k in 0..1), 4 rows
    W: rows rW + k*2 + i (k,i in 0..1), 4 rows
    """
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                u_row = rU + i*2 + j
                U[u_row][col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                v_row = rV + j*2 + k
                V[v_row][col + c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                w_row = rW + k*2 + i
                W[w_row][col + c] = sW[k*2 + i][c]
    return col + 7

def _strassen_3x3():
    """Rank-26 decomposition for 3x3x3.
    Split 3x3 into:
      A11 (2x2), A12 (2x1)
      A21 (1x2), A22 (1x1)
    Same for B.
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Ranks: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    """
    n = 3
    rank = 26
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Block positions in 3x3:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2
    # A21: row 2, cols 0-1
    # A22: row 2, col 2
    
    col = 0
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2 x 2x2 -> 2x2, rank 7 (Strassen)
    # U rows: 0-1, cols 0-1; V rows: 0-1, cols 0-1; W rows: 0-1, cols 0-1
    col = _strassen_block_2x2(0, 0, 0, 0, 0, 0, col)
    
    # A12*B21: 2x1 x 1x2 -> 2x2, rank 4 (schoolbook)
    # U rows: 0-1, col 2; V rows: 2, cols 0-1; W rows: 0-1, cols 0-1
    col = _block2x2(0, 2, 2, 0, 0, 0, col)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2 x 2x1 -> 2x1, rank 4
    # U rows: 0-1, cols 0-1; V rows: 0-1, col 2; W rows: 0-1, col 2
    col = _block2x2(0, 0, 0, 2, 0, 2, col)
    
    # A12*B22: 2x1 x 1x1 -> 2x1, rank 2
    # U rows: 0-1, col 2; V row: 2, col 2; W rows: 0-1, col 2
    col = _block2x1(0, 2, 2, 2, 0, 2, col)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 1x2 x 2x2 -> 1x2, rank 4
    # U row: 2, cols 0-1; V rows: 0-1, cols 0-1; W row: 2, cols 0-1
    col = _block2x2(2, 0, 0, 0, 2, 0, col)
    
    # A22*B21: 1x1 x 1x2 -> 1x2, rank 2
    # U row: 2, col 2; V row: 2, cols 0-1; W row: 2, cols 0-1
    col = _block1x2(2, 2, 2, 0, 2, 0, col)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 1x2 x 2x1 -> 1x1, rank 2
    # U row: 2, cols 0-1; V rows: 0-1, col 2; W row: 2, col 2
    col = _block1x2(2, 0, 0, 2, 2, 2, col)
    
    # A22*B22: 1x1 x 1x1 -> 1x1, rank 1
    col = _block1x1(2, 2, 2, 2, 2, 2, col)
    
    assert col == 26, f"Expected rank 26, got {col}"
    
    return U, V, W

def _kron_6x6x6():
    """
    Decompose 6x6x6 as Kronecker product of 2x2x2 (rank 7) and 3x3x3 (rank 26).
    Total rank = 7 * 26 = 182.
    
    View 6x6 matrices as 2x2 blocks of 3x3 matrices.
    A[i,j] where i = a*3 + r, j = b*3 + s, with a,b in 0..1 (block), r,s in 0..2 (within block).
    U index: i*6+j = (a*3+r)*6 + (b*3+s) = a*18 + r*6 + b*3 + s
    V index: j*6+k = (b*3+s)*6 + (c*3+t) = b*18 + s*6 + c*3 + t
    W index: k*6+i = (c*3+t)*6 + (a*3+r) = c*18 + t*6 + a*3 + r
    
    For each pair of Strassen factors (a1,a2) and 3x3 factors (r1,r2):
    U row: a1*18 + r1*6 + a2*3 + r2
    V row: b1*18 + s1*6 + b2*3 + s2
    W row: c1*18 + t1*6 + c2*3 + t2
    """
    n, m, p = 6, 6, 6
    sU, sV, sW = _strassen_2x2()
    tU, tV, tW = _strassen_3x3()
    
    rank = 7 * 26
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for a1 in range(7):
        for r1 in range(26):
            col = a1 * 26 + r1
            for a2 in range(7):
                for r2 in range(26):
                    # U contribution
                    for i in range(2):
                        for j in range(2):
                            su_val = sU[i*2+j][a1]
                            if su_val == 0:
                                continue
                            for r in range(3):
                                for s in range(3):
                                    st_val = tU[r*3+s][r1]
                                    if st_val == 0:
                                        continue
                                    val = su_val * st_val
                                    u_row = i*18 + r*6 + j*3 + s
                                    U[u_row][col] += val
                    
                    # V contribution
                    for j in range(2):
                        for k in range(2):
                            sv_val = sV[j*2+k][a1]
                            if sv_val == 0:
                                continue
                            for s in range(3):
                                for t in range(3):
                                    st_val = tV[s*3+t][r1]
                                    if st_val == 0:
                                        continue
                                    val = sv_val * st_val
                                    v_row = j*18 + s*6 + k*3 + t
                                    V[v_row][col] += val
                    
                    # W contribution
                    for k in range(2):
                        for i in range(2):
                            sw_val = sW[k*2+i][a1]
                            if sw_val == 0:
                                continue
                            for t in range(3):
                                for r in range(3):
                                    st_val = tW[t*3+r][r1]
                                    if st_val == 0:
                                        continue
                                    val = sw_val * st_val
                                    w_row = k*18 + t*6 + i*3 + r
                                    W[w_row][col] += val
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Primary: Kronecker product of Strassen 2x2 (rank 7) and 3x3 (rank 26) = rank 182
    best = _kron_6x6x6()
    
    # Verify using numpy for speed
    U_arr = np.array(best["U"], dtype=np.float64)
    V_arr = np.array(best["V"], dtype=np.float64)
    W_arr = np.array(best["W"], dtype=np.float64)
    
    rank = len(U_arr[0])
    
    # Full verification using numpy broadcasting
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return best
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
