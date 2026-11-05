import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build schoolbook rank-1 decomposition for n x m * m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 using Strassen (rank 7).
    Indices:
    U: (4, 7) for A (row-major 2x2)
    V: (4, 7) for B (row-major 2x2)
    W: (4, 7) for C (row-major 2x2)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 3, 3, 8
    # Rank 68 decomposition
    R = 68
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Part 1: A22 * B28 using Strassen for 4 blocks of 2x2
    # A22 is A[0:2, 0:2]
    # B28 is B[0:2, 0:8]
    # We split B28 into 4 blocks of 2x2: B[0:2, 2k:2k+2]
    
    U_s, V_s, W_s = _build_strassen_2x2_2x2_2x2()
    
    for k in range(4):
        # For each block k, we add 7 rank-1 terms
        for r in range(7):
            # Update U
            # A22 indices in U:
            # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 3, A[1,1] -> 4
            # U_s indices: 0, 1, 2, 3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            # Wait, U_s is 4x7. Rows 0..3.
            # 0: A00, 1: A01, 2: A10, 3: A11
            # Map to global U indices:
            # A00 (0,0) -> 0
            # A01 (0,1) -> 1
            # A10 (1,0) -> 3
            # A11 (1,1) -> 4
            
            U_indices = [0, 1, 3, 4]
            for i_local, i_global in enumerate(U_indices):
                U[i_global, col] += U_s[i_local, r]
                
            # Update V
            # B block k is B[0:2, 2k:2k+2]
            # V_s indices: 0: B00, 1: B01, 2: B10, 3: B11
            # Global V indices:
            # B00 (0, 2k) -> 0*8 + 2k = 2k
            # B01 (0, 2k+1) -> 0*8 + 2k+1 = 2k+1
            # B10 (1, 2k) -> 1*8 + 2k = 8+2k
            # B11 (1, 2k+1) -> 1*8 + 2k+1 = 8+2k+1
            
            V_indices = [2*k, 2*k+1, 8+2*k, 8+2*k+1]
            for i_local, i_global in enumerate(V_indices):
                V[i_global, col] += V_s[i_local, r]
                
            # Update W
            # C block k is C[0:2, 2k:2k+2]
            # W_s indices: 0: C00, 1: C01, 2: C10, 3: C11
            # Global W indices: W[k*n + i, r] where k is col, i is row.
            # C00 (0, 2k) -> (2k)*3 + 0 = 6k
            # C01 (0, 2k+1) -> (2k+1)*3 + 0 = 6k+3
            # C10 (1, 2k) -> (2k)*3 + 1 = 6k+1
            # C11 (1, 2k+1) -> (2k+1)*3 + 1 = 6k+4
            
            W_indices = [6*k, 6*k+3, 6*k+1, 6*k+4]
            for i_local, i_global in enumerate(W_indices):
                W[i_global, col] += W_s[i_local, r]
                
            col += 1
            
    # col should be 28 here
    
    # Part 2: A21 * B18
    # A21 is A[0:2, 2] -> indices 2, 5
    # B18 is B[2, 0:8] -> indices 16..23
    # C is C[0:2, 0:8]
    for i in range(2): # Rows of A21 (0, 1)
        u_idx = i * 3 + 2
        for c in range(8): # Cols of B18
            v_idx = 2 * 8 + c
            w_idx = c * 3 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # col should be 28 + 16 = 44
    
    # Part 3: A12 * B28
    # A12 is A[2, 0:2] -> indices 6, 7
    # B28 is B[0:2, 0:8]
    # C is C[2, 0:8]
    for j in range(2): # Cols of A12 / Rows of B28 (0, 1)
        u_idx = 2 * 3 + j
        for c in range(8): # Cols of B28
            v_idx = j * 8 + c
            w_idx = c * 3 + 2
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # col should be 44 + 16 = 60
    
    # Part 4: A11 * B18
    # A11 is A[2, 2] -> index 8
    # B18 is B[2, 0:8] -> indices 16..23
    # C is C[2, 0:8]
    for c in range(8):
        u_idx = 2 * 3 + 2
        v_idx = 2 * 8 + c
        w_idx = c * 3 + 2
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
        
    # col should be 60 + 8 = 68
    
    # Verify col == R
    if col != R:
        # Fallback to schoolbook if something went wrong
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
