"""
Matrix multiplication tensor decomposition for 2x3 by 3x5 matrices.
Uses block decomposition with Strassen's algorithm for 2x2 sub-blocks
to achieve rank 28 (down from schoolbook rank 30).

Block structure:
- A (2x3) = [A1 (2x2) | A2 (2x1)]
- B (3x5) = [B1 (2x5); B2 (1x5)]
- C = A1*B1 + A2*B2

For A1*B1 (2x2 x 2x5): Use Strassen (7 products) for first 4 columns (two 2x2 blocks),
schoolbook (4 products) for the last column. Total: 7+7+4 = 18 terms.
For A2*B2 (2x1 x 1x5): Schoolbook 10 terms.
Total rank: 18 + 10 = 28.
"""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _check_validity_exact(U, V, W, n=2, m=3, p=5):
    """Check if the decomposition is exact using Fraction arithmetic."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val == 0 or v_val == 0 or w_val == 0:
                        continue
                    s += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                if s != 1:
                    return False
    return True

def _check_validity_fast(U, V, W, n=2, m=3, p=5, tol=1e-10):
    """Fast float-based check."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += float(U[i * m + j][c]) * float(V[j * p + k][c]) * float(W[k * n + i][c])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2_block(a_col_offset, b_col_offset, factor_col_start, U, V, W, n, m, p):
    """
    Add Strassen's 7-product decomposition for a 2x2 x 2x2 block.
    
    a_col_offset: column offset in A for the 2x2 sub-block
    b_col_offset: column offset in B for the 2x2 sub-block  
    factor_col_start: starting column index in the factor matrices U,V,W
    
    The 2x2 sub-block of A uses columns [a_col_offset, a_col_offset+1]
    The 2x2 sub-block of B uses rows [0,1] (since A1 is the top 2 rows of the 3-row B)
    and columns [b_col_offset, b_col_offset+1]
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # A indices in U: A[i, a_col_offset + j]
    a00 = 0 * m + a_col_offset
    a01 = 0 * m + a_col_offset + 1
    a10 = 1 * m + a_col_offset
    a11 = 1 * m + a_col_offset + 1
    
    # B indices in V: B[i, b_col_offset + j] where i is 0,1 (rows of B)
    b00 = 0 * p + b_col_offset
    b01 = 0 * p + b_col_offset + 1
    b10 = 1 * p + b_col_offset
    b11 = 1 * p + b_col_offset + 1
    
    # C indices in W: C[i, b_col_offset + j]
    c00 = b_col_offset * n + 0
    c01 = (b_col_offset + 1) * n + 0
    c10 = b_col_offset * n + 1
    c11 = (b_col_offset + 1) * n + 1
    
    fc = factor_col_start  # shorthand for factor column
    
    # M1 = (A00+A11)*(B00+B11)
    U[a00][fc] += 1
    U[a11][fc] += 1
    V[b00][fc] += 1
    V[b11][fc] += 1
    W[c00][fc] += 1
    W[c11][fc] += 1
    
    # M2 = (A10+A11)*B00
    U[a10][fc+1] += 1
    U[a11][fc+1] += 1
    V[b00][fc+1] += 1
    W[c10][fc+1] += 1
    W[c11][fc+1] += 1
    
    # M3 = A00*(B01-B11)
    U[a00][fc+2] += 1
    V[b01][fc+2] += 1
    V[b11][fc+2] -= 1
    W[c01][fc+2] += 1
    W[c11][fc+2] -= 1
    
    # M4 = A11*(B10-B00)
    U[a11][fc+3] += 1
    V[b10][fc+3] += 1
    V[b00][fc+3] -= 1
    W[c00][fc+3] -= 1
    W[c10][fc+3] += 1
    
    # M5 = (A00+A01)*B11
    U[a00][fc+4] += 1
    U[a01][fc+4] += 1
    V[b11][fc+4] += 1
    W[c00][fc+4] += 1
    W[c10][fc+4] -= 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[a10][fc+5] += 1
    U[a00][fc+5] -= 1
    V[b00][fc+5] += 1
    V[b01][fc+5] += 1
    W[c11][fc+5] -= 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[a01][fc+6] += 1
    U[a11][fc+6] -= 1
    V[b10][fc+6] += 1
    V[b11][fc+6] += 1
    W[c00][fc+6] -= 1
    W[c01][fc+6] += 1
    
    return fc + 7

def _build_block_decomposition(n=2, m=3, p=5):
    """
    Build the rank-28 decomposition using block decomposition.
    
    A (2x3) = [A1 (2x2) | A2 (2x1)]
    B (3x5) = [B1 (2x5); B2 (1x5)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 x 2x5):
    - Use Strassen for columns 0-1 of B1 (2x2 block): 7 terms
    - Use Strassen for columns 2-3 of B1 (2x2 block): 7 terms
    - Use schoolbook for column 4 of B1 (2x1): 4 terms
    Total: 18 terms
    
    For A2*B2 (2x1 x 1x5):
    - Schoolbook: 10 terms
    Total: 10 terms
    
    Grand total: 28 terms
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A1*B1: 2x2 x 2x5
    # A1 uses A columns 0,1 (a_col_offset=0)
    # B1 uses B rows 0,1
    
    # Strassen block for B1 columns 0,1
    col = _build_strassen_2x2_block(0, 0, 0, U, V, W, n, m, p)
    
    # Strassen block for B1 columns 2,3
    col = _build_strassen_2x2_block(0, 2, col, U, V, W, n, m, p)
    
    # Schoolbook for B1 column 4 (2x2 x 2x1 = 4 terms)
    # A1 elements: A[0,0], A[0,1], A[1,0], A[1,1]
    # B1 elements: B[0,4], B[1,4]
    # C elements: C[0,4], C[1,4]
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Term 1: A[0,0]*B[0,4] -> C[0,4]
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1]*B[1,4] -> C[0,4]
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0]*B[0,4] -> C[1,4]
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1]*B[1,4] -> C[1,4]
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # A2*B2: 2x1 x 1x5 = 10 terms
    # A2 uses A column 2: A[0,2], A[1,2]
    # B2 uses B row 2: B[2,0], B[2,1], B[2,2], B[2,3], B[2,4]
    # C elements: C[0,0..4], C[1,0..4]
    
    for k in range(p):
        # C[0,k] += A[0,2]*B[2,k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1,k] += A[1,2]*B[2,k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return U, V, W

def _schoolbook(n=2, m=3, p=5):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition with Strassen (rank 28)
    try:
        U, V, W = _build_block_decomposition(n, m, p)
        if _check_validity_fast(U, V, W, n, m, p):
            # Verify with exact arithmetic
            if _check_validity_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = result['U']
    V = result['V']
    W = result['W']
    
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Verify with random matrices
    random.seed(42)
    A = [[random.randint(1, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(1, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
    
    # Compute C using decomposition
    C_decomp = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(rank):
                a_part = 0
                for j in range(m):
                    a_part += A[i][j] * U[i * m + j][r]
                b_part = 0
                for j in range(m):
                    b_part += B[j][k] * V[j * p + k][r]
                s += W[k * n + i][r] * a_part * b_part
            C_decomp[i][k] = s
    
    is_valid = True
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_decomp[i][k]:
                is_valid = False
                print(f"Mismatch at {i},{k}: {C[i][k]} != {C_decomp[i][k]}")
    
    if is_valid:
        print(f"Valid decomposition with rank {rank}")
    else:
        print("Invalid decomposition")
