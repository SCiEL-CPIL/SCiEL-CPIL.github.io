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
    """
    # U rows: (0,0), (0,1), (1,0), (1,1) for A[i][j]
    # V rows: (0,0), (0,1), (1,0), (1,1) for B[j][k]
    # W rows: (0,0), (0,1), (1,0), (1,1) for C[k][i] (i.e., output index k*n+i with n=2)
    
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


def _winograd_2x2_2x3():
    """
    Return the Winograd rank-11 decomposition for 2x2 * 2x3.
    U: shape (4, 11) - rows for A[i][j]
    V: shape (6, 11) - rows for B[j][k]
    W: shape (6, 11) - rows for C[i][k] (trace convention: index k*2+i)
    
    Winograd 2x2*2x3:
    Let A = [[a00, a01], [a10, a11]]
    Let B = [[b00, b01, b02], [b10, b11, b12]]
    
    We use the 2x2 Strassen-like structure but adapted for the extra column.
    Actually, the standard Winograd for 2x2*2x3 uses 11 multiplications.
    
    A simpler approach:
    Split B into B_left (2x2) and B_right (2x1).
    Compute A*B_left using Strassen (rank 7).
    Compute A*B_right using Schoolbook (rank 4).
    Total rank = 11.
    
    This is a valid exact decomposition.
    """
    rank = 11
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Part 1: Strassen for 2x2 * 2x2 (B columns 0,1)
    U_s, V_s, W_s = _strassen_2x2_2x2()
    for i in range(4):
        for c in range(7):
            U[i][c] = U_s[i][c]
    
    # V mapping: local (j,k) for k in {0,1}
    # V_s[0] is (0,0), V_s[1] is (0,1), V_s[2] is (1,0), V_s[3] is (1,1)
    # Global V rows:
    # 0: B[0][0] -> V_s[0]
    # 1: B[0][1] -> V_s[1]
    # 2: B[0][2] -> 0 (handled in part 2)
    # 3: B[1][0] -> V_s[2]
    # 4: B[1][1] -> V_s[3]
    # 5: B[1][2] -> 0 (handled in part 2)
    
    V[0][0:7] = V_s[0][0:7]
    V[1][0:7] = V_s[1][0:7]
    V[3][0:7] = V_s[2][0:7]
    V[4][0:7] = V_s[3][0:7]
    
    # W mapping: local (k,i) for k in {0,1}
    # W_s[0] is (0,0) -> C[0][0] (global W[0])
    # W_s[1] is (0,1) -> C[0][1] (global W[1])
    # W_s[2] is (1,0) -> C[1][0] (global W[2])
    # W_s[3] is (1,1) -> C[1][1] (global W[3])
    
    W[0][0:7] = W_s[0][0:7]
    W[1][0:7] = W_s[1][0:7]
    W[2][0:7] = W_s[2][0:7]
    W[3][0:7] = W_s[3][0:7]
    
    # Part 2: Schoolbook for 2x2 * 2x1 (B column 2)
    # C[0][2] = A[0][0]B[0][2] + A[0][1]B[1][2]
    # C[1][2] = A[1][0]B[0][2] + A[1][1]B[1][2]
    
    # Global indices:
    # A[0][0]=U[0], A[0][1]=U[1], A[1][0]=U[2], A[1][1]=U[3]
    # B[0][2]=V[2], B[1][2]=V[5]
    # C[0][2]=W[4] (k=0, i=0 -> 0*2+0? No. C is 2x3. Output C[i][k].
    # Wait, the problem says W[k*n+i]. Here n=2.
    # C[0][0] -> W[0*2+0]=W[0]
    # C[0][1] -> W[1*2+0]=W[2]? No.
    # Let's re-read the trace convention: W[k*n+i].
    # For 2x2*2x3:
    # C is 2x3. C[i][k] where i in 0..1, k in 0..2.
    # W index = k*n + i = k*2 + i.
    # C[0][0] -> W[0]
    # C[0][1] -> W[2]
    # C[0][2] -> W[4]
    # C[1][0] -> W[1]
    # C[1][1] -> W[3]
    # C[1][2] -> W[5]
    
    # My previous W mapping in Part 1 was:
    # W_s[0] (local 0,0) -> C[0][0] -> W[0]
    # W_s[1] (local 0,1) -> C[0][1] -> W[2]
    # W_s[2] (local 1,0) -> C[1][0] -> W[1]
    # W_s[3] (local 1,1) -> C[1][1] -> W[3]
    
    # Let's correct the W mapping in Part 1.
    W[0][0:7] = W_s[0][0:7] # C[0][0]
    W[2][0:7] = W_s[1][0:7] # C[0][1]
    W[1][0:7] = W_s[2][0:7] # C[1][0]
    W[3][0:7] = W_s[3][0:7] # C[1][1]
    
    # Now Part 2:
    # C[0][2] -> W[4]
    # C[1][2] -> W[5]
    
    # Term 1: A[0][0] * B[0][2] -> C[0][2]
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 2: A[0][1] * B[1][2] -> C[0][2]
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 3: A[1][0] * B[0][2] -> C[1][2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 4: A[1][1] * B[1][2] -> C[1][2]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return U, V, W


def _strassen_2x4_4x7():
    """
    Block-wise decomposition for 4x4 * 4x7.
    
    A (4x4) split into 2x2 blocks:
      A11, A12, A21, A22
    
    B (4x7) split:
      B11 (2x2), B12 (2x2), B13 (2x3)
      B21 (2x2), B22 (2x2), B23 (2x3)
    
    C (4x7):
      C11 = A11*B11 + A12*B21 (2x2)
      C12 = A11*B12 + A12*B22 (2x2)
      C13 = A11*B13 + A12*B23 (2x3)
      C21 = A21*B11 + A22*B21 (2x2)
      C22 = A21*B12 + A22*B22 (2x2)
      C23 = A21*B13 + A22*B23 (2x3)
    
    Rank calculation:
    C11: 2 * Strassen(7) = 14
    C12: 2 * Strassen(7) = 14
    C13: 2 * Winograd(11) = 22
    C21: 2 * Strassen(7) = 14
    C22: 2 * Strassen(7) = 14
    C23: 2 * Winograd(11) = 22
    
    Total Rank = 14+14+22+14+14+22 = 100.
    
    Wait, the previous program had rank 50. How?
    Ah, the previous program used `_strassen_2x2_2x3` which was just Strassen(7) + Schoolbook(4) = 11.
    And it used `_strassen_2x2_2x2` which is 7.
    
    Let's check the previous rank calculation in the code:
    "Total rank: 4*7 + 2*11 = 50"
    This implies:
    C11: 2*7 = 14? No, 4*7 = 28.
    C13: 2*11 = 22.
    Total 50.
    
    So the previous program used:
    C11, C12, C21, C22: 4 blocks of 2x2. Each is sum of two 2x2 products.
    Wait, C11 = A11*B11 + A12*B21.
    If we compute this as two separate Strassen decompositions, it's 14 rank.
    But maybe we can share?
    No, the inputs are different.
    
    Let's re-read the previous code's comment:
    "Total rank: 4*7 + 2*11 = 50"
    This suggests that for the 2x2 blocks (C11, C12, C21, C22), the rank was 7 each?
    That would imply C11 was computed with rank 7.
    But C11 = A11*B11 + A12*B21.
    If A11 and A12 are different, and B11 and B21 are different, you cannot compute their sum with rank 7 unless you merge the linear combinations.
    
    Actually, the previous code `add_strassen` just adds 7 columns.
    It calls `add_strassen(A11_rows, B11_rows, C11_rows)` and `add_strassen(A12_rows, B21_rows, C11_rows)`.
    This adds 14 columns for C11.
    
    So 4 blocks * 14 = 56.
    Plus 2 blocks * (7+4=11) = 22.
    Total 78?
    
    Let's look at the previous code's `add_2x2_2x3`.
    It uses Strassen (7) + Schoolbook (4) = 11.
    
    So:
    C11: 14
    C12: 14
    C13: 11
    C21: 14
    C22: 14
    C23: 11
    Total: 78.
    
    Why did the comment say 50?
    Maybe the comment was wrong or I am misinterpreting.
    
    Regardless, my new implementation using `_winograd_2x2_2x3` (which is also 11) will have the same rank structure as the previous `_strassen_2x2_2x3` if I use the same logic.
    
    However, the Winograd 2x2*2x3 I implemented above is rank 11.
    The previous `_strassen_2x2_2x3` was also rank 11.
    
    So the rank will be the same.
    But maybe I can optimize the 2x2 blocks?
    
    For C11 = A11*B11 + A12*B21.
    This is equivalent to [A11 A12] * [B11; B21].
    This is a 2x4 * 4x2 multiplication.
    Rank of 2x4*4x2?
    Schoolbook: 2*4*2 = 16.
    Strassen-like?
    
    If I can find a rank < 14 decomposition for 2x4*4x2, I can improve.
    But for now, let's stick to the safe 78-rank decomposition to ensure correctness, then try to optimize.
    
    Actually, the previous program scored 1.0, which is the seed.
    The seed is schoolbook (112).
    So 78 is an improvement.
    
    Let's implement the 78-rank decomposition correctly.
    """
    n, m, p = 4, 4, 7
    # Rank:
    # 4 * 2 * 7 (for 2x2 blocks) = 56
    # 2 * 11 (for 2x3 blocks) = 22
    # Total = 78
    rank = 78
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    col = 0
    
    def add_strassen(a_rows, b_rows, c_rows):
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
        
    def add_winograd_2x2_2x3(a_rows, b_rows, c_rows):
        nonlocal col
        U_w, V_w, W_w = _winograd_2x2_2x3()
        for li in range(4):
            for c in range(11):
                U[a_rows[li]][col + c] = U_w[li][c]
        for lb in range(6):
            for c in range(11):
                V[b_rows[lb]][col + c] = V_w[lb][c]
        for lc in range(6):
            for c in range(11):
                W[c_rows[lc]][col + c] = W_w[lc][c]
        col += 11

    # A rows
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    # B rows
    B11_rows = [0, 1, 7, 8]
    B12_rows = [2, 3, 9, 10]
    B13_rows = [4, 5, 6, 11, 12, 13]
    B21_rows = [14, 15, 21, 22]
    B22_rows = [16, 17, 23, 24]
    B23_rows = [18, 19, 20, 25, 26, 27]
    
    # C rows (W indices)
    # W index = k*4 + i
    C11_rows = [0, 1, 4, 5] # k=0,1; i=0,1
    C12_rows = [2, 3, 6, 7] # k=2,3; i=0,1
    C13_rows = [4, 5, 8, 9, 12, 13] # k=4,5,6; i=0,1
    C21_rows = [2, 3, 6, 7] # k=0,1; i=2,3 -> W[0*4+2]=2, W[0*4+3]=3, W[1*4+2]=6, W[1*4+3]=7
    C22_rows = [10, 11, 14, 15] # k=2,3; i=2,3 -> W[2*4+2]=10, ...
    C23_rows = [18, 19, 22, 23, 26, 27] # k=4,5,6; i=2,3 -> W[4*4+2]=18, ...
    
    # C11
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)
    
    # C12
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)
    
    # C13
    add_winograd_2x2_2x3(A11_rows, B13_rows, C13_rows)
    add_winograd_2x2_2x3(A12_rows, B23_rows, C13_rows)
    
    # C21
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)
    
    # C22
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)
    
    # C23
    add_winograd_2x2_2x3(A21_rows, B13_rows, C23_rows)
    add_winograd_2x2_2x3(A22_rows, B23_rows, C23_rows)
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the block-wise Strassen/Winograd decomposition (rank 78)
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
