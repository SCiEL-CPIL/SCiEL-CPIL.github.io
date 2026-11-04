"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def _strassen_2x2_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    U: shape (4, 7) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (4, 7) - rows indexed by (j,k) for j,k in {0,1}
    W: shape (4, 7) - rows indexed by (k,i) for k,i in {0,1}
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U rows: (0,0), (0,1), (1,0), (1,1) for A[i][j]
    # V rows: (0,0), (0,1), (1,0), (1,1) for B[j][k]
    # W rows: (0,0), (0,1), (1,0), (1,1) for C[k][i] (i.e., output index k*n+i with n=2)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # A11 = (0,0), A22 = (1,1), B11 = (0,0), B22 = (1,1)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1   # C[0,0] += M1
    U[1][0] = 1; V[1][0] = 1; W[1][0] = 1   # C[1,1] += M1 (W row for (k=1,i=1) is index 1*2+1=3? No wait)
    
    # Let me re-index carefully.
    # For 2x2*2x2: n=2, m=2, p=2
    # U row index: i*m + j = i*2 + j
    # V row index: j*p + k = j*2 + k
    # W row index: k*n + i = k*2 + i
    
    # So:
    # U[0] = A[0][0], U[1] = A[0][1], U[2] = A[1][0], U[3] = A[1][1]
    # V[0] = B[0][0], V[1] = B[0][1], V[2] = B[1][0], V[3] = B[1][1]
    # W[0] = C[0][0], W[1] = C[0][1], W[2] = C[1][0], W[3] = C[1][1]
    
    # Reset
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00=U[0], A11=U[3], B00=V[0], B11=V[3]
    # C00=W[0], C11=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10=U[2], A11=U[3], B00=V[0]
    # C10=W[2], C11=W[3]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    # A00=U[0], B10=V[2], B11=V[3]
    # C01=W[1], C11=W[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    # A11=U[3], B01=V[1], B00=V[0]
    # C00=W[0], C10=W[2]
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    # A00=U[0], A01=U[1], B11=V[3]
    # C00=W[0], C01=W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10=U[2], A00=U[0], B00=V[0], B01=V[1]
    # C11=W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01=U[1], A11=U[3], B10=V[2], B11=V[3]
    # C00=W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_2x2_2x4():
    """
    Strassen-based decomposition for 2x2 * 2x4.
    Split B into two 2x2 blocks: B = [B_left | B_right]
    C = [C_left | C_right] where C_left = A*B_left (2x2), C_right = A*B_right (2x2)
    
    We can compute both using Strassen, but we need to be careful.
    Actually, let's use the standard approach:
    A is 2x2, B is 2x4 = [B1 | B2] where B1, B2 are 2x2.
    C = A*B = [A*B1 | A*B2]
    
    For A*B1: use Strassen (7 mults)
    For A*B2: use Strassen (7 mults)
    Total: 14 mults, but we can potentially share some.
    
    Actually, a better approach: use the fact that we can write
    A*B = (A11+A22, A21+A22, ...) applied to each block of B.
    
    Let me just use 2 independent Strassen decompositions for the two 2x2 subproblems.
    That gives rank 14 for 2x2*2x4.
    
    Alternatively, we can use a more efficient construction.
    
    For now, let's use 2 Strassen blocks: rank 14.
    """
    # U: shape (4, 14) - rows for A[i][j], i,j in {0,1}
    # V: shape (8, 14) - rows for B[j][k], j in {0,1}, k in {0,1,2,3}
    # W: shape (8, 14) - rows for C[k][i], k in {0,1,2,3}, i in {0,1}
    
    # B is 2x4: B[0][0], B[0][1], B[0][2], B[0][3], B[1][0], B[1][1], B[1][2], B[1][3]
    # V row index: j*4 + k
    # V[0]=B[0][0], V[1]=B[0][1], V[2]=B[0][2], V[3]=B[0][3]
    # V[4]=B[1][0], V[5]=B[1][1], V[6]=B[1][2], V[7]=B[1][3]
    
    # C is 2x4: C[0][0], C[0][1], C[0][2], C[0][3], C[1][0], C[1][1], C[1][2], C[1][3]
    # W row index: k*2 + i (since p=4, n=2)
    # W[0]=C[0][0], W[1]=C[0][1], W[2]=C[0][2], W[3]=C[0][3]
    # W[4]=C[1][0], W[5]=C[1][1], W[6]=C[1][2], W[7]=C[1][3]
    
    U, V_left, W_left = _strassen_2x2_2x2()  # For B_left = B[:, 0:2]
    U_right, V_right, W_right = _strassen_2x2_2x2()  # For B_right = B[:, 2:4]
    
    rank = 14
    U_full = [[0]*rank for _ in range(4)]
    V_full = [[0]*rank for _ in range(8)]
    W_full = [[0]*rank for _ in range(8)]
    
    # Left half (columns 0,1 of B)
    for i in range(4):
        for c in range(7):
            U_full[i][c] = U[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_full[j*4 + k][c] = V_left[j*2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_full[k*2 + i][c] = W_left[k*2 + i][c]
    
    # Right half (columns 2,3 of B)
    for i in range(4):
        for c in range(7):
            U_full[i][c + 7] = U_right[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_full[j*4 + (k+2)][c + 7] = V_right[j*2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_full[(k+2)*2 + i][c + 7] = W_right[k*2 + i][c]
    
    return U_full, V_full, W_full


def _strassen_2x4_4x7():
    """
    Block-wise Strassen decomposition for 4x4 * 4x7.
    
    A (4x4) split into 2x2 blocks:
      A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    
    B (4x7) split:
      B11 = B[0:2,0:2], B12 = B[0:2,2:4], B13 = B[0:2,4:7]
      B21 = B[2:4,0:2], B22 = B[2:4,2:4], B23 = B[2:4,4:7]
    
    C (4x7):
      C11 = A11*B11 + A12*B21  (2x2)  -> Strassen rank 7
      C12 = A11*B12 + A12*B22  (2x2)  -> Strassen rank 7
      C13 = A11*B13 + A12*B23  (2x3)  -> rank 11
      C21 = A21*B11 + A22*B21  (2x2)  -> Strassen rank 7
      C22 = A21*B12 + A22*B22  (2x2)  -> Strassen rank 7
      C23 = A21*B13 + A22*B23  (2x3)  -> rank 11
    
    Total rank: 4*7 + 2*11 = 50
    """
    n, m, p = 4, 4, 7
    rank = 50
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    col = 0
    
    # U row for A[i][j] = i*4 + j
    # V row for B[j][k] = j*7 + k
    # W row for C[i][k] = k*4 + i  (trace convention W[k*n+i])
    
    # For a 2x2 A block with top-left at (ai, aj):
    #   local (0,0)->U[ai*4+aj], (0,1)->U[ai*4+aj+1], (1,0)->U[(ai+1)*4+aj], (1,1)->U[(ai+1)*4+aj+1]
    # For a 2x2 B block with top-left at (bj, bk):
    #   local (0,0)->V[bj*7+bk], (0,1)->V[bj*7+bk+1], (1,0)->V[(bj+1)*7+bk], (1,1)->V[(bj+1)*7+bk+1]
    # For a 2x2 C block with top-left at (ci, ck):
    #   local (0,0)->W[ck*4+ci], (0,1)->W[ck*4+ci+1], (1,0)->W[(ck+1)*4+ci], (1,1)->W[(ck+1)*4+ci+1]
    
    def add_strassen(a_rows, b_rows, c_rows):
        """Add Strassen 2x2*2x2 using explicit row lists."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col + c] = U_s[li][c]
        for lb in range(4):
            for c in range(7):
                V[b_rows[lb]][col + c] = V_s[lb][c]
        for lc in range(4):
            for c in range(7):
                W[c_rows[lc]][col + c] = W_s[lc][c]
        col += 7
    
    def add_2x2_2x3(a_rows, b_rows, c_rows):
        """
        Add 2x2*2x3 decomposition.
        a_rows: 4 U-row indices for the 2x2 A block (local order 00,01,10,11)
        b_rows: 6 V-row indices for the 2x3 B block (row-major: b00,b01,b02,b10,b11,b12)
        c_rows: 6 W-row indices for the 2x3 C block (row-major: c00,c01,c02,c10,c11,c12)
        """
        nonlocal col
        # Part 1: A * B[:,0:2] using Strassen
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col + c] = U_s[li][c]
        # B local: (0,0)->b_rows[0], (0,1)->b_rows[1], (1,0)->b_rows[3], (1,1)->b_rows[4]
        V[b_rows[0]][col+0] = V_s[0][0]; V[b_rows[0]][col+1] = V_s[0][1]; V[b_rows[0]][col+2] = V_s[0][2]; V[b_rows[0]][col+3] = V_s[0][3]; V[b_rows[0]][col+4] = V_s[0][4]; V[b_rows[0]][col+5] = V_s[0][5]; V[b_rows[0]][col+6] = V_s[0][6]
        V[b_rows[1]][col+0] = V_s[1][0]; V[b_rows[1]][col+1] = V_s[1][1]; V[b_rows[1]][col+2] = V_s[1][2]; V[b_rows[1]][col+3] = V_s[1][3]; V[b_rows[1]][col+4] = V_s[1][4]; V[b_rows[1]][col+5] = V_s[1][5]; V[b_rows[1]][col+6] = V_s[1][6]
        V[b_rows[3]][col+0] = V_s[2][0]; V[b_rows[3]][col+1] = V_s[2][1]; V[b_rows[3]][col+2] = V_s[2][2]; V[b_rows[3]][col+3] = V_s[2][3]; V[b_rows[3]][col+4] = V_s[2][4]; V[b_rows[3]][col+5] = V_s[2][5]; V[b_rows[3]][col+6] = V_s[2][6]
        V[b_rows[4]][col+0] = V_s[3][0]; V[b_rows[4]][col+1] = V_s[3][1]; V[b_rows[4]][col+2] = V_s[3][2]; V[b_rows[4]][col+3] = V_s[3][3]; V[b_rows[4]][col+4] = V_s[3][4]; V[b_rows[4]][col+5] = V_s[3][5]; V[b_rows[4]][col+6] = V_s[3][6]
        # C local: (0,0)->c_rows[0], (0,1)->c_rows[1], (1,0)->c_rows[3], (1,1)->c_rows[4]
        W[c_rows[0]][col+0] = W_s[0][0]; W[c_rows[0]][col+1] = W_s[0][1]; W[c_rows[0]][col+2] = W_s[0][2]; W[c_rows[0]][col+3] = W_s[0][3]; W[c_rows[0]][col+4] = W_s[0][4]; W[c_rows[0]][col+5] = W_s[0][5]; W[c_rows[0]][col+6] = W_s[0][6]
        W[c_rows[1]][col+0] = W_s[1][0]; W[c_rows[1]][col+1] = W_s[1][1]; W[c_rows[1]][col+2] = W_s[1][2]; W[c_rows[1]][col+3] = W_s[1][3]; W[c_rows[1]][col+4] = W_s[1][4]; W[c_rows[1]][col+5] = W_s[1][5]; W[c_rows[1]][col+6] = W_s[1][6]
        W[c_rows[3]][col+0] = W_s[2][0]; W[c_rows[3]][col+1] = W_s[2][1]; W[c_rows[3]][col+2] = W_s[2][2]; W[c_rows[3]][col+3] = W_s[2][3]; W[c_rows[3]][col+4] = W_s[2][4]; W[c_rows[3]][col+5] = W_s[2][5]; W[c_rows[3]][col+6] = W_s[2][6]
        W[c_rows[4]][col+0] = W_s[3][0]; W[c_rows[4]][col+1] = W_s[3][1]; W[c_rows[4]][col+2] = W_s[3][2]; W[c_rows[4]][col+3] = W_s[3][3]; W[c_rows[4]][col+4] = W_s[3][4]; W[c_rows[4]][col+5] = W_s[3][5]; W[c_rows[4]][col+6] = W_s[3][6]
        col += 7
        
        # Part 2: A * B[:,2] (2x2 * 2x1) schoolbook
        # C[i][2] = A[i][0]*B[0][2] + A[i][1]*B[1][2]
        # a_rows: [a00, a01, a10, a11]
        # b_rows[2] = B[0][2], b_rows[5] = B[1][2]
        # c_rows[2] = C[0][2], c_rows[5] = C[1][2]
        for i_local in range(2):
            for j_local in range(2):
                a_idx = a_rows[i_local*2 + j_local]
                b_idx = b_rows[j_local*3 + 2]
                c_idx = c_rows[2*3 + i_local]
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    # A11: i=0,1; j=0,1 -> U rows [0,1,4,5]
    # A12: i=0,1; j=2,3 -> U rows [2,3,6,7]
    # A21: i=2,3; j=0,1 -> U rows [8,9,12,13]
    # A22: i=2,3; j=2,3 -> U rows [10,11,14,15]
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    # B11: j=0,1; k=0,1 -> V rows [0,1,7,8]
    # B12: j=0,1; k=2,3 -> V rows [2,3,9,10]
    # B13: j=0,1; k=4,5,6 -> V rows [4,5,6,11,12,13]
    # B21: j=2,3; k=0,1 -> V rows [14,15,21,22]
    # B22: j=2,3; k=2,3 -> V rows [16,17,23,24]
    # B23: j=2,3; k=4,5,6 -> V rows [18,19,20,25,26,27]
    B11_rows = [0, 1, 7, 8]
    B12_rows = [2, 3, 9, 10]
    B13_rows = [4, 5, 6, 11, 12, 13]
    B21_rows = [14, 15, 21, 22]
    B22_rows = [16, 17, 23, 24]
    B23_rows = [18, 19, 20, 25, 26, 27]
    
    # C11: i=0,1; k=0,1 -> W rows [0,1,4,5]
    # C12: i=0,1; k=2,3 -> W rows [2,3,6,7]
    # C13: i=0,1; k=4,5,6 -> W rows [4,5,8,9,12,13]
    # C21: i=2,3; k=0,1 -> W rows [2,3,6,7]
    # C22: i=2,3; k=2,3 -> W rows [10,11,14,15]
    # C23: i=2,3; k=4,5,6 -> W rows [18,19,22,23,26,27]
    C11_rows = [0, 1, 4, 5]
    C12_rows = [2, 3, 6, 7]
    C13_rows = [4, 5, 8, 9, 12, 13]
    C21_rows = [2, 3, 6, 7]
    C22_rows = [10, 11, 14, 15]
    C23_rows = [18, 19, 22, 23, 26, 27]
    
    # C11 = A11*B11 + A12*B21
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)
    
    # C12 = A11*B12 + A12*B22
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)
    
    # C13 = A11*B13 + A12*B23
    add_2x2_2x3(A11_rows, B13_rows, C13_rows)
    add_2x2_2x3(A12_rows, B23_rows, C13_rows)
    
    # C21 = A21*B11 + A22*B21
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)
    
    # C22 = A21*B12 + A22*B22
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)
    
    # C23 = A21*B13 + A22*B23
    add_2x2_2x3(A21_rows, B13_rows, C23_rows)
    add_2x2_2x3(A22_rows, B23_rows, C23_rows)
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the block-wise Strassen decomposition (rank 50)
    try:
        U, V, W = _strassen_2x4_4x7()
        result = {"U": U, "V": V, "W": W}
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
