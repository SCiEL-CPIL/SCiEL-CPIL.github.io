import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1 for all i,j,k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U_rows, V_rows, W_rows, i0, i1, j0, j1, k0, k1, n, m, p, col_offset):
    """
    Add a Strassen 2x2x2 decomposition for sub-blocks.
    i0,i1: row indices in A (and C)
    j0,j1: column indices in A (and row indices in B)
    k0,k1: column indices in B (and C)
    Returns new col_offset.
    """
    U = U_rows
    V = V_rows
    W = W_rows
    
    # Strassen 7 products for 2x2x2
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
    
    col = col_offset
    
    # M1
    U[i0*m+j0][col] += 1
    U[i1*m+j1][col] += 1
    V[j0*p+k0][col] += 1
    V[j1*p+k1][col] += 1
    W[k0*n+i0][col] += 1
    W[k1*n+i1][col] += 1
    col += 1
    
    # M2
    U[i1*m+j0][col] += 1
    U[i1*m+j1][col] += 1
    V[j0*p+k0][col] += 1
    W[k0*n+i1][col] += 1
    col += 1
    
    # M3
    U[i0*m+j0][col] += 1
    V[j0*p+k1][col] += 1
    V[j1*p+k1][col] += -1
    W[k1*n+i0][col] += 1
    col += 1
    
    # M4
    U[i1*m+j1][col] += 1
    V[j1*p+k0][col] += 1
    V[j0*p+k0][col] += -1
    W[k0*n+i1][col] += 1
    col += 1
    
    # M5
    U[i0*m+j0][col] += 1
    U[i0*m+j1][col] += 1
    V[j1*p+k1][col] += 1
    W[k1*n+i0][col] += -1
    col += 1
    
    # M6
    U[i1*m+j0][col] += 1
    U[i0*m+j0][col] += -1
    V[j0*p+k0][col] += 1
    V[j0*p+k1][col] += 1
    W[k0*n+i1][col] += 1
    col += 1
    
    # M7
    U[i0*m+j1][col] += 1
    U[i1*m+j1][col] += -1
    V[j1*p+k0][col] += 1
    V[j1*p+k1][col] += 1
    W[k1*n+i0][col] += 1
    col += 1
    
    return col

def _block_decomposition(n, m, p):
    """
    Block-partitioned decomposition using Strassen on 2x2x2 sub-blocks.
    For 3x5x5, we can place 4 non-overlapping 2x2x2 Strassen blocks.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Identify non-overlapping 2x2x2 blocks
    # Rows of A: 0,1,2 -> pairs: (0,1)
    # Cols of A / Rows of B: 0,1,2,3,4 -> pairs: (0,1), (2,3), (4,?) - can't pair 4
    # Cols of B / Cols of C: 0,1,2,3,4 -> pairs: (0,1), (2,3), (4,?) - can't pair 4
    
    # We can place Strassen blocks where both dimensions have even sizes.
    # Use rows (0,1), cols (0,1) and (2,3) for A/B interface, and output cols (0,1) and (2,3)
    
    # Block 1: rows 0,1; cols 0,1; outputs 0,1
    col = _strassen_2x2x2(U, V, W, 0, 1, 0, 1, 0, 1, n, m, p, col)
    
    # Block 2: rows 0,1; cols 0,1; outputs 2,3
    col = _strassen_2x2x2(U, V, W, 0, 1, 0, 1, 2, 3, n, m, p, col)
    
    # Block 3: rows 0,1; cols 2,3; outputs 0,1
    col = _strassen_2x2x2(U, V, W, 0, 1, 2, 3, 0, 1, n, m, p, col)
    
    # Block 4: rows 0,1; cols 2,3; outputs 2,3
    col = _strassen_2x2x2(U, V, W, 0, 1, 2, 3, 2, 3, n, m, p, col)
    
    # Now handle remaining terms with schoolbook
    # We've covered: rows 0,1; cols 0,1,2,3; outputs 0,1,2,3
    # Remaining:
    # - Row 2 of A with all cols and all outputs
    # - Rows 0,1 with col 4 and all outputs
    # - Rows 0,1 with cols 0,1,2,3 and output 4
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is already covered by Strassen
                # Strassen covers: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3]:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try block decomposition
    result = _block_decomposition(n, m, p)
    U, V, W = result['U'], result['V'], result['W']
    current_rank = len(U[0])
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        elapsed = time.time() - start_time
        if elapsed < 140.0:
            return result
    
    # Fallback to trivial
    return _trivial_decomposition(n, m, p)

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
