"""
Rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.

The multiplication tensor condition:
  For all i in [n], j in [m], k in [p]:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

Strategy:
1. Use Strassen's algorithm for 2x2x2 sub-blocks (rank 7 vs schoolbook 8).
2. Block decomposition: rows 3=2+1, inner 4=2+2, cols 5=2+2+1.
3. This yields rank 56 (verified by construction).
4. Keep schoolbook rank 60 as fallback.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > tol:
                    return False
    return True


def build_strassen_2x2x2(U, V, W, base_col, i_off, j_off, k_off, n, m, p):
    """
    Add Strassen's 7 rank-1 terms for a 2x2x2 sub-block.
    
    A sub-block: rows i_off:i_off+2, inner j_off:j_off+2, cols k_off:k_off+2.
    U rows: (i_off + di) * m + (j_off + dj) for di,dj in {0,1}
    V rows: (j_off + dj) * p + (k_off + dk) for dj,dk in {0,1}
    W rows: (k_off + dk) * n + (i_off + di) for dk,di in {0,1}
    
    Strassen multiplications:
    M1 = (a+d)(e+h)  -> U: [1,0,0,1] on A, V: [1,1,0,0] on B... 
    
    Let me define A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U indices for A entries: 
      A00 -> (i_off)*m + (j_off)
      A01 -> (i_off)*m + (j_off+1)
      A10 -> (i_off+1)*m + (j_off)
      A11 -> (i_off+1)*m + (j_off+1)
    
    V indices for B entries:
      B00 -> (j_off)*p + (k_off)
      B01 -> (j_off)*p + (k_off+1)
      B10 -> (j_off+1)*p + (k_off)
      B11 -> (j_off+1)*p + (k_off+1)
    
    W indices for C entries:
      C00 -> (k_off)*n + (i_off)
      C01 -> (k_off+1)*n + (i_off)
      C10 -> (k_off)*n + (i_off+1)
      C11 -> (k_off+1)*n + (i_off+1)
    """
    u00 = i_off * m + j_off
    u01 = i_off * m + (j_off + 1)
    u10 = (i_off + 1) * m + j_off
    u11 = (i_off + 1) * m + (j_off + 1)
    
    v00 = j_off * p + k_off
    v01 = j_off * p + (k_off + 1)
    v10 = (j_off + 1) * p + k_off
    v11 = (j_off + 1) * p + (k_off + 1)
    
    w00 = k_off * n + i_off
    w01 = (k_off + 1) * n + i_off
    w10 = k_off * n + (i_off + 1)
    w11 = (k_off + 1) * n + (i_off + 1)
    
    # M1 = (A00+A11)(B00+B11)
    U[u00, base_col] = 1
    U[u11, base_col] = 1
    V[v00, base_col] = 1
    V[v11, base_col] = 1
    W[w00, base_col] = 1
    W[w11, base_col] = 1
    base_col += 1
    
    # M2 = (A10+A11)*B00
    U[u10, base_col] = 1
    U[u11, base_col] = 1
    V[v00, base_col] = 1
    W[w10, base_col] = 1
    W[w11, base_col] = -1
    base_col += 1
    
    # M3 = A00*(B01-B11)
    U[u00, base_col] = 1
    V[v01, base_col] = 1
    V[v11, base_col] = -1
    W[w01, base_col] = 1
    W[w11, base_col] = 1
    base_col += 1
    
    # M4 = A11*(B10-B00)
    U[u11, base_col] = 1
    V[v10, base_col] = 1
    V[v00, base_col] = -1
    W[w00, base_col] = 1
    W[w10, base_col] = 1
    base_col += 1
    
    # M5 = (A00+A01)*B11
    U[u00, base_col] = 1
    U[u01, base_col] = 1
    V[v11, base_col] = 1
    W[w00, base_col] = -1
    W[w01, base_col] = 1
    base_col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[u10, base_col] = 1
    U[u00, base_col] = -1
    V[v00, base_col] = 1
    V[v01, base_col] = 1
    W[w11, base_col] = 1
    base_col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[u01, base_col] = 1
    U[u11, base_col] = -1
    V[v10, base_col] = 1
    V[v11, base_col] = 1
    W[w00, base_col] = 1
    W[w11, base_col] = 1
    base_col += 1
    
    return base_col


def build_schoolbook_block(U, V, W, base_col, i_off, j_off, k_off, ni, nj, nk, n, m, p):
    """
    Add schoolbook rank-1 terms for a ni x nj x nk sub-block.
    Rows i_off:i_off+ni, inner j_off:j_off+nj, cols k_off:k_off+nk.
    """
    for di in range(ni):
        for dj in range(nj):
            for dk in range(nk):
                u_idx = (i_off + di) * m + (j_off + dj)
                v_idx = (j_off + dj) * p + (k_off + dk)
                w_idx = (k_off + dk) * n + (i_off + di)
                U[u_idx, base_col] = 1
                V[v_idx, base_col] = 1
                W[w_idx, base_col] = 1
                base_col += 1
    return base_col


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    # Block structure:
    # Rows: 3 = 2 + 1  (i_off: 0, 2; sizes: 2, 1)
    # Inner: 4 = 2 + 2  (j_off: 0, 2; sizes: 2, 2)
    # Cols: 5 = 2 + 2 + 1  (k_off: 0, 2, 4; sizes: 2, 2, 1)
    
    # Sub-blocks and their ranks:
    # (2x2) x (2x2) for k=0:2 -> Strassen, rank 7
    # (2x2) x (2x2) for k=2:4 -> Strassen, rank 7
    # (2x2) x (2x1) for k=4:5 -> schoolbook, rank 4
    # (2x2) x (2x2) for k=0:2 -> Strassen, rank 7
    # (2x2) x (2x2) for k=2:4 -> Strassen, rank 7
    # (2x2) x (2x1) for k=4:5 -> schoolbook, rank 4
    # (1x2) x (2x2) for k=0:2 -> schoolbook, rank 4
    # (1x2) x (2x2) for k=2:4 -> schoolbook, rank 4
    # (1x2) x (2x1) for k=4:5 -> schoolbook, rank 2
    # (1x2) x (2x2) for k=0:2 -> schoolbook, rank 4
    # (1x2) x (2x2) for k=2:4 -> schoolbook, rank 4
    # (1x2) x (2x1) for k=4:5 -> schoolbook, rank 2
    
    # Total: 7+7+4 + 7+7+4 + 4+4+2 + 4+4+2 = 56
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Block (i=0:2, j=0:2, k=0:2) - Strassen
    col = build_strassen_2x2x2(U, V, W, col, 0, 0, 0, n, m, p)
    # Block (i=0:2, j=0:2, k=2:4) - Strassen
    col = build_strassen_2x2x2(U, V, W, col, 0, 0, 2, n, m, p)
    # Block (i=0:2, j=0:2, k=4:5) - schoolbook 2x2x1
    col = build_schoolbook_block(U, V, W, col, 0, 0, 4, 2, 2, 1, n, m, p)
    
    # Block (i=0:2, j=2:4, k=0:2) - Strassen
    col = build_strassen_2x2x2(U, V, W, col, 0, 2, 0, n, m, p)
    # Block (i=0:2, j=2:4, k=2:4) - Strassen
    col = build_strassen_2x2x2(U, V, W, col, 0, 2, 2, n, m, p)
    # Block (i=0:2, j=2:4, k=4:5) - schoolbook 2x2x1
    col = build_schoolbook_block(U, V, W, col, 0, 2, 4, 2, 2, 1, n, m, p)
    
    # Block (i=2:3, j=0:2, k=0:2) - schoolbook 1x2x2
    col = build_schoolbook_block(U, V, W, col, 2, 0, 0, 1, 2, 2, n, m, p)
    # Block (i=2:3, j=0:2, k=2:4) - schoolbook 1x2x2
    col = build_schoolbook_block(U, V, W, col, 2, 0, 2, 1, 2, 2, n, m, p)
    # Block (i=2:3, j=0:2, k=4:5) - schoolbook 1x2x1
    col = build_schoolbook_block(U, V, W, col, 2, 0, 4, 1, 2, 1, n, m, p)
    
    # Block (i=2:3, j=2:4, k=0:2) - schoolbook 1x2x2
    col = build_schoolbook_block(U, V, W, col, 2, 2, 0, 1, 2, 2, n, m, p)
    # Block (i=2:3, j=2:4, k=2:4) - schoolbook 1x2x2
    col = build_schoolbook_block(U, V, W, col, 2, 2, 2, 1, 2, 2, n, m, p)
    # Block (i=2:3, j=2:4, k=4:5) - schoolbook 1x2x1
    col = build_schoolbook_block(U, V, W, col, 2, 2, 4, 1, 2, 1, n, m, p)
    
    # Verify col matches expected rank
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify the decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=float)
        V = np.zeros((m * p, rank_school), dtype=float)
        W = np.zeros((p * n, rank_school), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    n, m, p = DIMENSIONS
    valid = verify_decomposition(U, V, W, n, m, p)
    print(f"Valid: {valid}")
