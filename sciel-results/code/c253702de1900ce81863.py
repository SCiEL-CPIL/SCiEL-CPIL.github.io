"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that non-tensor entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c_idx // n
                i2 = c_idx % n
                if i == i2 and j == j2 and k == k2:
                    continue  # This is a valid tensor entry, already checked
                s = 0.0
                for cc in range(rank):
                    s += U[a, cc] * V[b, cc] * W[c_idx, cc]
                if abs(s) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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

def _add_schoolbook_block(U, V, W, n, m, p, i0, j0, k0, start_col):
    """Add schoolbook decomposition for sub-block at offsets i0,j0,k0."""
    col = start_col
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i0 + i) * m + (j0 + j)][col] = 1
                V[(j0 + j) * p + (k0 + k)][col] = 1
                W[(k0 + k) * n + (i0 + i)][col] = 1
                col += 1
    return col

def _add_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0, start_col):
    """
    Add Strassen decomposition for 2x2x2 block at offsets i0,j0,k0.
    Uses 7 rank terms.
    
    Standard Strassen:
    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22)B11
    P3 = A11(B21 - B22)
    P4 = A22(B12 - B11)
    P5 = (A11 + A12)B22
    P6 = (A21 - A11)(B11 + B12)
    P7 = (A12 - A22)(B21 + B22)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    """
    col = start_col
    
    # P1 = (A00+A11)(B00+B11)
    U[i0 * m + j0][col] = 1      # A00
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + k0][col] = 1      # B00
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = 1      # C00
    W[(k0+1) * n + (i0+1)][col] = 1 # C11
    col += 1
    
    # P2 = (A10+A11)B00
    U[(i0+1) * m + j0][col] = 1  # A10
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + k0][col] = 1      # B00
    W[k0 * n + (i0+1)][col] = 1  # C10
    col += 1
    
    # P3 = A00(B10-B11)
    U[i0 * m + j0][col] = 1      # A00
    V[(j0+1) * p + k0][col] = 1  # B10
    V[(j0+1) * p + (k0+1)][col] = -1 # B11
    W[(k0+1) * n + i0][col] = 1  # C01
    col += 1
    
    # P4 = A11(B01-B00)
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + (k0+1)][col] = 1    # B01
    V[j0 * p + k0][col] = -1       # B00
    W[k0 * n + i0][col] = 1        # C00
    W[k0 * n + (i0+1)][col] = 1    # C10
    col += 1
    
    # P5 = (A00+A01)B11
    U[i0 * m + j0][col] = 1       # A00
    U[i0 * m + (j0+1)][col] = 1   # A01
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = -1      # C00
    W[(k0+1) * n + i0][col] = 1   # C01
    col += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[(i0+1) * m + j0][col] = 1   # A10
    U[i0 * m + j0][col] = -1      # A00
    V[j0 * p + k0][col] = 1       # B00
    V[j0 * p + (k0+1)][col] = 1   # B01
    W[(k0+1) * n + (i0+1)][col] = 1 # C11
    col += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[i0 * m + (j0+1)][col] = 1   # A01
    U[(i0+1) * m + (j0+1)][col] = -1 # A11
    V[(j0+1) * p + k0][col] = 1   # B10
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = 1       # C00
    col += 1
    
    return col

def _construct_rank34(n=3, m=3, p=4):
    """
    Construct a rank-34 decomposition using Strassen for 2x2x2 blocks.
    
    Block decomposition:
    - A[0:2,0:2] * B[0:2,0:2]: Strassen 2x2x2 = 7 terms
    - A[0:2,0:2] * B[0:2,2:4]: Strassen 2x2x2 = 7 terms
    - A[0:2,2] * B[2,0:2]: schoolbook 2x1x2 = 4 terms
    - A[0:2,2] * B[2,2:4]: schoolbook 2x1x2 = 4 terms
    - A[2,0:2] * B[0:2,0:2]: schoolbook 1x2x2 = 4 terms
    - A[2,0:2] * B[0:2,2:4]: schoolbook 1x2x2 = 4 terms
    - A[2,2] * B[2,0:2]: schoolbook 1x1x2 = 2 terms
    - A[2,2] * B[2,2:4]: schoolbook 1x1x2 = 2 terms
    Total: 34
    """
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 2, col)
    
    # Block 3: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    col = _add_schoolbook_block(U, V, W, 2, 1, 2, 0, 2, 0, col)
    
    # Block 4: A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
    col = _add_schoolbook_block(U, V, W, 2, 1, 2, 0, 2, 2, col)
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    col = _add_schoolbook_block(U, V, W, 1, 2, 2, 2, 0, 0, col)
    
    # Block 6: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    col = _add_schoolbook_block(U, V, W, 1, 2, 2, 2, 0, 2, col)
    
    # Block 7: A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    col = _add_schoolbook_block(U, V, W, 1, 1, 2, 2, 2, 0, col)
    
    # Block 8: A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    col = _add_schoolbook_block(U, V, W, 1, 1, 2, 2, 2, 2, col)
    
    if col != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {col}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank-34 construction first
    if time.time() < deadline - 10:
        result = _construct_rank34(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
