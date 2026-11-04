"""Rewritten for 3x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 8]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _add_strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    """Add a 2x2x2 Strassen block at A[i0:i0+2, j0:j0+2] x B[j0:j0+2, k0:k0+2].
    Returns new offset."""
    # Strassen M1..M7 for 2x2x2
    # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U entries (rows = A indices):
    #   row i0, j0 -> A00
    #   row i0, j0+1 -> A01
    #   row i0+1, j0 -> A10
    #   row i0+1, j0+1 -> A11
    # V entries (rows = B indices):
    #   row j0, k0 -> B00
    #   row j0, k0+1 -> B01
    #   row j0+1, k0 -> B10
    #   row j0+1, k0+1 -> B11
    # W entries (rows = C indices, k*n + i):
    #   row k0, i0 -> C00
    #   row k0, i0+1 -> C01
    #   row k0+1, i0 -> C10
    #   row k0+1, i0+1 -> C11
    
    A00 = i0 * m + j0
    A01 = i0 * m + (j0 + 1)
    A10 = (i0 + 1) * m + j0
    A11 = (i0 + 1) * m + (j0 + 1)
    B00 = j0 * p + k0
    B01 = j0 * p + (k0 + 1)
    B10 = (j0 + 1) * p + k0
    B11 = (j0 + 1) * p + (k0 + 1)
    C00 = k0 * n + i0
    C01 = k0 * n + (i0 + 1)
    C10 = (k0 + 1) * n + i0
    C11 = (k0 + 1) * n + (i0 + 1)
    
    terms = [
        # (U_indices, signs, V_indices, signs, W_indices, signs)
        # M1: (A00+A11)*(B00+B11) -> C00+=1, C11+=1, C01-=1, C10-=1
        # Wait, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6, C00 = M1+M4-M5+M7
        # So M1 contributes: C00 += 1, C11 += 1
        ([A00, A11], [1, 1], [B00, B11], [1, 1], [C00, C11], [1, 1]),
        # M2: (A10+A11)*B00 -> C10 += 1, C11 -= 1
        ([A10, A11], [1, 1], [B00], [1], [C10, C11], [1, -1]),
        # M3: A00*(B01-B11) -> C01 += 1, C11 += 1
        ([A00], [1], [B01, B11], [1, -1], [C01, C11], [1, 1]),
        # M4: A11*(B10-B00) -> C00 += 1, C10 += 1
        ([A11], [1], [B10, B00], [1, -1], [C00, C10], [1, 1]),
        # M5: (A00+A01)*B11 -> C00 -= 1, C01 += 1
        ([A00, A01], [1, 1], [B11], [1], [C00, C01], [-1, 1]),
        # M6: (A10-A00)*(B00+B01) -> C11 += 1
        ([A10, A00], [1, -1], [B00, B01], [1, 1], [C11], [1]),
        # M7: (A01-A11)*(B10+B11) -> C00 += 1
        ([A01, A11], [1, -1], [B10, B11], [1, 1], [C00], [1]),
    ]
    
    for u_idx, u_sign, v_idx, v_sign, w_idx, w_sign in terms:
        U[u_idx, offset] += np.array(u_sign, dtype=np.int8)
        V[v_idx, offset] += np.array(v_sign, dtype=np.int8)
        W[w_idx, offset] += np.array(w_sign, dtype=np.int8)
        offset += 1
    return offset

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    R = n * m * p
    
    # We'll build a new decomposition with rank 68
    # 4 non-overlapping 2x2x2 blocks in A (i in {0,1}, j in {0,1})
    # For each block, B has 2 rows (j in {0,1}) and 2 columns (k pairs)
    # The 8 columns of B split into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    
    U_new = np.zeros((n * m, 68), dtype=np.int8)
    V_new = np.zeros((m * p, 68), dtype=np.int8)
    W_new = np.zeros((p * n, 68), dtype=np.int8)
    
    col = 0
    
    # For each k-pair (k0 = 0, 2, 4, 6), add Strassen block for A[i in {0,1}, j in {0,1}]
    for k0 in range(0, p, 2):
        col = _add_strassen_2x2x2(U_new, V_new, W_new, 0, 0, k0, col)
        # col should be k0//2 * 7 + 7 = (k0//2 + 1) * 7
        # For k0=0: 7, k0=2: 14, k0=4: 21, k0=6: 28
    
    # Now add schoolbook terms for the remaining cases:
    # i=2 or j=2 (but not both in the Strassen-covered region)
    # Strassen covers: i in {0,1}, j in {0,1}
    # Remaining: (i=2, j=0), (i=2, j=1), (i=0, j=2), (i=1, j=2), (i=2, j=2)
    # For each of these 5 (i,j) pairs, and each k in 0..7, add a schoolbook term
    
    for i in range(n):
        for j in range(m):
            if i in (0, 1) and j in (0, 1):
                continue  # Already covered by Strassen
            for k in range(p):
                U_new[i * m + j, col] = 1
                V_new[j * p + k, col] = 1
                W_new[k * n + i, col] = 1
                col += 1
    
    # col should be 28 + 5 * 8 = 28 + 40 = 68
    assert col == 68, f"Expected 68, got {col}"
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
