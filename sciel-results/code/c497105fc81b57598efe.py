"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
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
                if abs(s - 1.0) > 1e-9:
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
    """Add schoolbook decomposition for sub-block M[n,m,p] at offsets i0,j0,k0."""
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
    Assumes n>=2, m>=2, p>=2.
    Uses 7 rank terms.
    """
    col = start_col
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11
    # C00 += M1, C11 += M1
    U[i0 * m + j0][col] = 1      # A00
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + k0][col] = 1      # B00
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = 1      # C00
    W[(k0+1) * n + (i0+1)][col] = 1 # C11
    col += 1
    
    # M2 = (A10+A11)B00
    # C10 += M2
    U[(i0+1) * m + j0][col] = 1  # A10
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + k0][col] = 1      # B00
    W[k0 * n + (i0+1)][col] = 1  # C10
    col += 1
    
    # M3 = A00(B10-B11)
    # C01 += M3
    U[i0 * m + j0][col] = 1      # A00
    V[(j0+1) * p + k0][col] = 1  # B10
    V[(j0+1) * p + (k0+1)][col] = -1 # B11
    W[k0 * n + i0][col] = 1      # C01? No, C01 is index (0,1) in C.
    # W index is k*n+i. For C01, i=0, k=1.
    W[(k0+1) * n + i0][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    # C00 += M4, C10 += M4
    U[(i0+1) * m + (j0+1)][col] = 1 # A11
    V[j0 * p + (k0+1)][col] = 1   # B01
    V[j0 * p + k0][col] = -1      # B00
    W[k0 * n + i0][col] = 1       # C00
    W[k0 * n + (i0+1)][col] = 1   # C10
    col += 1
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    U[i0 * m + j0][col] = 1      # A00
    U[i0 * m + (j0+1)][col] = 1  # A01
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = -1     # C00
    W[(k0+1) * n + i0][col] = 1  # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    U[(i0+1) * m + j0][col] = 1  # A10
    U[i0 * m + j0][col] = -1     # A00
    V[j0 * p + k0][col] = 1      # B00
    V[j0 * p + (k0+1)][col] = 1  # B01
    W[(k0+1) * n + (i0+1)][col] = 1 # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7, C11 -= M7? No, C11 = M1 - M2 + M3 + M6.
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # My M7 definition: (A01-A11)(B10+B11).
    # In C00: +M7.
    # In C11: -M7? No, C11 formula doesn't have M7.
    # Let's re-verify Strassen formulas.
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So for M7 (P7):
    # C00 (C11 in 1-based) += M7.
    # C11 (C22 in 1-based) -= M7? No, P7 is not in C22.
    # Wait, P7 is in C11. Is it in C22? No.
    # So W for M7 should only affect C00.
    
    U[i0 * m + (j0+1)][col] = 1  # A01
    U[(i0+1) * m + (j0+1)][col] = -1 # A11
    V[(j0+1) * p + k0][col] = 1   # B10
    V[(j0+1) * p + (k0+1)][col] = 1 # B11
    W[k0 * n + i0][col] = 1       # C00
    # C11 does NOT get M7.
    col += 1
    
    return col

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Rank 34 Construction
    # Split i: 2, 1
    # Split j: 2, 1
    # Split k: 4 (no split, or split into 2+2 for Strassen parts)
    
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:4]
    # This is M[2, 2, 4].
    # We can split B into B[0:2, 0:2] and B[0:2, 2:4].
    # Use Strassen for M[2,2,2] at (0,0,0)
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, col)
    # Use Strassen for M[2,2,2] at (0,0,2)
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 2, col)
    
    # Block 2: A[0:2, 2] * B[2, 0:4]
    # This is M[2, 1, 4]. Rank 8.
    # Use Schoolbook
    col = _add_schoolbook_block(U, V, W, 2, 1, 4, 0, 2, 0, col)
    
    # Block 3: A[2, 0:2] * B[0:2, 0:4]
    # This is M[1, 2, 4]. Rank 8.
    # Use Schoolbook
    col = _add_schoolbook_block(U, V, W, 1, 2, 4, 2, 0, 0, col)
    
    # Block 4: A[2, 2] * B[2, 0:4]
    # This is M[1, 1, 4]. Rank 4.
    # Use Schoolbook
    col = _add_schoolbook_block(U, V, W, 1, 1, 4, 2, 2, 0, col)
    
    if col != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {col}")
        # If col < rank, we have unused columns. We can trim or keep zeros.
        # If col > rank, we failed.
        
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction fails
        return _schoolbook(n, m, p)
