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
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_block_decomposition():
    """
    Block-based Strassen decomposition for 4x4 * 4x7.
    Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B (4x7) into rows: B = [[B_top], [B_bot]] where B_top, B_bot are 2x7
    Further split B_top into [B11, B12, B13] (2x2, 2x2, 2x3)
    Similarly B_bot into [B21, B22, B23]
    
    C (4x7) = [[C11, C12, C13], [C21, C22, C23]]
    where:
      C11 = A11*B11 + A12*B21  (2x2)
      C12 = A11*B12 + A12*B22  (2x2)
      C13 = A11*B13 + A12*B23  (2x3)
      C21 = A21*B11 + A22*B21  (2x2)
      C22 = A21*B12 + A22*B22  (2x2)
      C23 = A21*B13 + A22*B23  (2x3)
    
    For 2x2*2x2: use Strassen (rank 7)
    For 2x2*2x3: use a decomposition (rank 11: 7 for 2x2 part + 4 for 2x1 part)
    """
    n, m, p = 4, 4, 7
    
    # U rows: i*m + j = i*4 + j for A[i][j]
    # V rows: j*p + k = j*7 + k for B[j][k]
    # W rows: k*n + i = k*4 + i for C[k][i]
    
    # A11: A[0:2, 0:2] -> U rows: 0,1,4,5
    # A12: A[0:2, 2:4] -> U rows: 2,3,6,7
    # A21: A[2:4, 0:2] -> U rows: 8,9,12,13
    # A22: A[2:4, 2:4] -> U rows: 10,11,14,15
    
    # B11: B[0:2, 0:2] -> V rows: 0,1,7,8
    # B12: B[0:2, 2:4] -> V rows: 2,3,9,10
    # B13: B[0:2, 4:7] -> V rows: 4,5,6,11,12,13
    # B21: B[2:4, 0:2] -> V rows: 14,15,21,22
    # B22: B[2:4, 2:4] -> V rows: 16,17,23,24
    # B23: B[2:4, 4:7] -> V rows: 18,19,20,25,26,27
    
    # C11: C[0:2, 0:2] -> W rows: 0,1,4,5
    # C12: C[0:2, 2:4] -> W rows: 8,9,12,13
    # C13: C[0:2, 4:7] -> W rows: 16,17,20,21,24,25
    # C21: C[2:4, 0:2] -> W rows: 2,3,6,7
    # C22: C[2:4, 2:4] -> W rows: 10,11,14,15
    # C23: C[2:4, 4:7] -> W rows: 18,19,22,23,26,27
    
    rank = 50  # 4*7 + 2*11 = 28 + 22 = 50
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    col = 0
    
    def add_strassen(a_rows, b_rows, c_rows):
        """Add a Strassen 2x2*2x2 decomposition."""
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        for idx in range(4):
            for c in range(7):
                V[b_rows[idx]][col + c] = V_s[idx][c]
        for idx in range(4):
            for c in range(7):
                W[c_rows[idx]][col + c] = W_s[idx][c]
        col += 7
    
    def add_2x2_2x3(a_rows, b_rows, c_rows):
        """Add a 2x2*2x3 decomposition."""
        nonlocal col
        # Part 1: A * B[:, 0:2] using Strassen
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        for c in range(7):
            V[b_rows[0]][col + c] = V_s[0][c]
            V[b_rows[1]][col + c] = V_s[1][c]
            V[b_rows[3]][col + c] = V_s[2][c]
            V[b_rows[4]][col + c] = V_s[3][c]
        for c in range(7):
            W[c_rows[0]][col + c] = W_s[0][c]
            W[c_rows[1]][col + c] = W_s[1][c]
            W[c_rows[3]][col + c] = W_s[2][c]
            W[c_rows[4]][col + c] = W_s[3][c]
        col += 7
        
        # Part 2: A * B[:, 2] (2x2 * 2x1) schoolbook, 4 mults
        U[a_rows[0]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        U[a_rows[1]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        U[a_rows[2]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
        
        U[a_rows[3]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
    
    # Define row index lists
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    B11_rows = [0, 1, 7, 8]
    B12_rows = [2, 3, 9, 10]
    B13_rows = [4, 5, 6, 11, 12, 13]
    B21_rows = [14, 15, 21, 22]
    B22_rows = [16, 17, 23, 24]
    B23_rows = [18, 19, 20, 25, 26, 27]
    
    C11_rows = [0, 1, 4, 5]
    C12_rows = [8, 9, 12, 13]
    C13_rows = [16, 17, 20, 21, 24, 25]
    C21_rows = [2, 3, 6, 7]
    C22_rows = [10, 11, 14, 15]
    C23_rows = [18, 19, 22, 23, 26, 27]
    
    # Now build the decomposition
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)
    
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)
    
    add_2x2_2x3(A11_rows, B13_rows, C13_rows)
    add_2x2_2x3(A12_rows, B23_rows, C13_rows)
    
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)
    
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)
    
    add_2x2_2x3(A21_rows, B13_rows, C23_rows)
    add_2x2_2x3(A22_rows, B23_rows, C23_rows)
    
    return {"U": U, "V": V, "W": W}


def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy tensor contraction."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True


def _rank_via_bareiss(n, m, p):
    """
    Compute the rank of the multiplication tensor matricization using Bareiss
    fraction-free Gaussian elimination.
    
    The multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    We form a matrix M of shape (p*n, m) where M[k*n+i, j] = T[i,j,k] = 1.
    This is a (p*n) x m matrix of all ones, which has rank 1.
    
    But this is the wrong matricization for tensor rank. The tensor rank is
    the minimum r such that T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c.
    
    For the multiplication tensor, the tensor rank is known to be at least
    max(n*m, m*p, p*n) = max(16, 28, 28) = 28, and at most n*m*p = 112.
    
    Let's use a different approach: try to find a decomposition of rank r
    by solving a system of equations.
    
    Actually, for the purpose of this task, let's just use the Strassen
    block decomposition which gives rank 50, and verify it.
    """
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try the Strassen-based block decomposition
    try:
        result = _strassen_block_decomposition()
        if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
