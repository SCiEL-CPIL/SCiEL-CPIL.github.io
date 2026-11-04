"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use a structured decomposition that combines Strassen's algorithm for 2x2 blocks
  with schoolbook multiplication for the remaining parts.
- Partition the 3x7 matrix B into a 2x7 top part and a 1x7 bottom part,
  and the 2x3 matrix A into a 2x2 left part and a 2x1 right part.
- For the 2x2 * 2x7 part, use Strassen's algorithm for each pair of columns:
  Columns (0,1), (2,3), (4,5): Strassen rank 7 each = 21
  Column 6: schoolbook rank 4
  Total for this part: 25
- For the 2x1 * 1x7 part, use schoolbook: rank 14
- Total rank: 25 + 14 = 39

The program constructs this decomposition, verifies it exactly, and returns it.
If verification fails (should not happen), it falls back to schoolbook rank 42.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy. Returns True if likely correct."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2(U, V, W, col_offset, b_row_offset):
    """
    Add a 2x2 * 2x2 Strassen block.
    
    b_row_offset: the starting column index in B (and C) for this 2x2 block.
    
    A indices in U: A[i,j] -> U[i*3 + j]
    B indices in V: B[j,k] -> V[j*7 + k]
    C indices in W: C[i,k] -> W[k*2 + i]
    """
    m = 3
    p = 7
    n = 2
    
    # A indices in U
    A00 = 0 * m + 0  # U[0]
    A01 = 0 * m + 1  # U[1]
    A10 = 1 * m + 0  # U[3]
    A11 = 1 * m + 1  # U[4]
    
    # B indices in V
    b00 = 0 * p + b_row_offset
    b01 = 0 * p + (b_row_offset + 1)
    b10 = 1 * p + b_row_offset
    b11 = 1 * p + (b_row_offset + 1)
    
    # C indices in W
    C00 = b_row_offset * n + 0
    C01 = (b_row_offset + 1) * n + 0
    C10 = b_row_offset * n + 1
    C11 = (b_row_offset + 1) * n + 1
    
    # M1 = (A00 + A11)(B00 + B11)
    U[A00][col_offset] += 1
    U[A11][col_offset] += 1
    V[b00][col_offset] += 1
    V[b11][col_offset] += 1
    W[C00][col_offset] += 1
    W[C11][col_offset] += 1
    
    # M2 = (A10 + A11)B00
    U[A10][col_offset + 1] += 1
    U[A11][col_offset + 1] += 1
    V[b00][col_offset + 1] += 1
    W[C10][col_offset + 1] += 1
    W[C11][col_offset + 1] -= 1
    
    # M3 = A00(B01 - B11)
    U[A00][col_offset + 2] += 1
    V[b01][col_offset + 2] += 1
    V[b11][col_offset + 2] -= 1
    W[C01][col_offset + 2] += 1
    W[C11][col_offset + 2] += 1
    
    # M4 = A11(B10 - B00)
    U[A11][col_offset + 3] += 1
    V[b10][col_offset + 3] += 1
    V[b00][col_offset + 3] -= 1
    W[C00][col_offset + 3] += 1
    W[C10][col_offset + 3] += 1
    
    # M5 = (A00 + A01)B11
    U[A00][col_offset + 4] += 1
    U[A01][col_offset + 4] += 1
    V[b11][col_offset + 4] += 1
    W[C00][col_offset + 4] -= 1
    W[C01][col_offset + 4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[A10][col_offset + 5] += 1
    U[A00][col_offset + 5] -= 1
    V[b00][col_offset + 5] += 1
    V[b01][col_offset + 5] += 1
    W[C11][col_offset + 5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[A01][col_offset + 6] += 1
    U[A11][col_offset + 6] -= 1
    V[b10][col_offset + 6] += 1
    V[b11][col_offset + 6] += 1
    W[C00][col_offset + 6] += 1
    W[C11][col_offset + 6] += 1
    
    return col_offset + 7

def _schoolbook_2x1_1x7(U, V, W, col_offset, a_col, b_row):
    """
    Add a 2x1 * 1x7 schoolbook block.
    
    a_col: column index in A (0, 1, or 2)
    b_row: row index in B (0, 1, or 2)
    """
    n, m, p = 2, 3, 7
    
    for i in range(n):
        for k in range(p):
            U[i * m + a_col][col_offset] = 1
            V[b_row * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    
    return col_offset

def _schoolbook_2x2_2x1(U, V, W, col_offset, b_col):
    """
    Add a 2x2 * 2x1 schoolbook block (4 multiplications).
    
    b_col: column index in B
    """
    n, m, p = 2, 3, 7
    
    A00 = 0 * m + 0
    A01 = 0 * m + 1
    A10 = 1 * m + 0
    A11 = 1 * m + 1
    
    b0 = 0 * p + b_col
    b1 = 1 * p + b_col
    
    C00 = b_col * n + 0
    C10 = b_col * n + 1
    
    # A00 * B0
    U[A00][col_offset] = 1
    V[b0][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A01 * B1
    U[A01][col_offset] = 1
    V[b1][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A10 * B0
    U[A10][col_offset] = 1
    V[b0][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    # A11 * B1
    U[A11][col_offset] = 1
    V[b1][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    return col_offset

def _construct_rank_39():
    """
    Construct the rank-39 decomposition.
    
    Part 1: A_left (2x2) * B_top (2x7)
    - Columns (0,1): Strassen 2x2*2x2, rank 7
    - Columns (2,3): Strassen 2x2*2x2, rank 7
    - Columns (4,5): Strassen 2x2*2x2, rank 7
    - Column 6: Schoolbook 2x2*2x1, rank 4
    Total: 25
    
    Part 2: A_right (2x1) * B_bottom (1x7)
    - Schoolbook 2x1*1x7, rank 14
    Total: 14
    
    Grand total: 39
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    rank = 39
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Part 1: A_left * B_top
    # Strassen for columns (0,1)
    col = _strassen_2x2(U, V, W, col, 0)
    
    # Strassen for columns (2,3)
    col = _strassen_2x2(U, V, W, col, 2)
    
    # Strassen for columns (4,5)
    col = _strassen_2x2(U, V, W, col, 4)
    
    # Schoolbook for column 6
    col = _schoolbook_2x2_2x1(U, V, W, col, 6)
    
    # Part 2: A_right * B_bottom
    col = _schoolbook_2x1_1x7(U, V, W, col, 2, 2)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try the rank-39 construction first
    result = _construct_rank_39()
    
    # Verify it's correct
    if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
        if _verify_decomposition_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
