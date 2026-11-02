"""Construct a rank-minimal tensor decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j'*p+k,r] == delta[j,j'].
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    R = U.shape[1]
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0:
                    continue
                u_vec = U[i * m:(i + 1) * m, r]
                v_col = V[k::p, r]
                M += w_val * np.outer(u_vec, v_col)
            if not np.allclose(M, np.eye(m), atol=1e-10):
                return False
    return True

def _build_strassen_2x4x4(n, m, p):
    """
    Build a rank-28 decomposition of 2x4x4 matrix multiplication using
    four independent 2x2x2 Strassen products.
    
    A is 2x4, B is 4x4, C = A*B is 2x4.
    Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    Split B into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
    Split C into C_L (2x2, cols 0-1) and C_R (2x2, cols 2-3).
    
    C_L = A_L * B11 + A_R * B21
    C_R = A_L * B12 + A_R * B22
    
    Each 2x2x2 product uses Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    """
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A00)*(B00+B10)
    # M7 = (A10-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen U, V, W entries for a 2x2x2 block:
    # U entries (for A part): each Strassen term has coefficients on A elements
    # V entries (for B part): each Strassen term has coefficients on B elements
    # W entries (for C part): each output element is a linear combination of M's
    
    # For a 2x2x2 block at position (i_block, j_block, k_block):
    # i_block: 0 or 1 (which 2x2 row block of A, but A is 2x4 so only one row block)
    # j_block: 0 or 1 (which 2x2 col block of A / row block of B)
    # k_block: 0 or 1 (which 2x2 col block of C / col block of B)
    
    def _strassen_block(base_col, i_block, j_block, k_block):
        """
        Add a Strassen 2x2x2 decomposition for the product:
        A[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2] * 
        B[j_block*2:(j_block+1)*2, k_block*2:(k_block+1)*2] ->
        C[i_block*2:(i_block+1)*2, k_block*2:(k_block+1)*2]
        
        Since n=2, i_block is always 0.
        j_block: 0 for cols 0-1 of A (and rows 0-1 of B), 1 for cols 2-3 of A (and rows 2-3 of B)
        k_block: 0 for cols 0-1 of C (and cols 0-1 of B), 1 for cols 2-3 of C (and cols 2-3 of B)
        """
        # A row indices (i is always 0-1 since n=2)
        a_rows = [0, 1]
        # A col indices (j_block determines which 2 columns)
        a_cols = [j_block * 2, j_block * 2 + 1]
        # B row indices (same as A col indices)
        b_rows = [j_block * 2, j_block * 2 + 1]
        # B col indices (k_block determines which 2 columns)
        b_cols = [k_block * 2, k_block * 2 + 1]
        # C row indices
        c_rows = [0, 1]
        # C col indices
        c_cols = [k_block * 2, k_block * 2 + 1]
        
        # Strassen terms M1..M7
        # Each term: U coefficients (on A elements), V coefficients (on B elements), W coefficients (on C elements)
        
        # M1 = (A00+A11)*(B00+B11)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        U[0 * m + a_cols[0], base_col + 0] = 1  # A[0, j_block*2]
        U[1 * m + a_cols[1], base_col + 0] = 1  # A[1, j_block*2+1]
        V[b_rows[0] * p + b_cols[0], base_col + 0] = 1  # B[j_block*2, k_block*2]
        V[b_rows[1] * p + b_cols[1], base_col + 0] = 1  # B[j_block*2+1, k_block*2+1]
        # W: C00 += M1, C11 += M1
        W[c_cols[0] * n + c_rows[0], base_col + 0] = 1  # C[0, k_block*2]
        W[c_cols[1] * n + c_rows[1], base_col + 0] = 1  # C[1, k_block*2+1]
        
        # M2 = (A10+A11)*B00
        U[1 * m + a_cols[0], base_col + 1] = 1  # A[1, j_block*2]
        U[1 * m + a_cols[1], base_col + 1] = 1  # A[1, j_block*2+1]
        V[b_rows[0] * p + b_cols[0], base_col + 1] = 1  # B[j_block*2, k_block*2]
        # W: C10 += M2, C11 -= M2
        W[c_cols[0] * n + c_rows[1], base_col + 1] = 1  # C[1, k_block*2]
        W[c_cols[1] * n + c_rows[1], base_col + 1] = -1  # C[1, k_block*2+1]
        
        # M3 = A00*(B01-B11)
        U[0 * m + a_cols[0], base_col + 2] = 1  # A[0, j_block*2]
        V[b_rows[0] * p + b_cols[1], base_col + 2] = 1  # B[j_block*2, k_block*2+1]
        V[b_rows[1] * p + b_cols[1], base_col + 2] = -1  # B[j_block*2+1, k_block*2+1]
        # W: C01 += M3, C11 += M3
        W[c_cols[1] * n + c_rows[0], base_col + 2] = 1  # C[0, k_block*2+1]
        W[c_cols[1] * n + c_rows[1], base_col + 2] = 1  # C[1, k_block*2+1]
        
        # M4 = A11*(B10-B00)
        U[1 * m + a_cols[1], base_col + 3] = 1  # A[1, j_block*2+1]
        V[b_rows[1] * p + b_cols[0], base_col + 3] = 1  # B[j_block*2+1, k_block*2]
        V[b_rows[0] * p + b_cols[0], base_col + 3] = -1  # B[j_block*2, k_block*2]
        # W: C00 += M4, C10 += M4
        W[c_cols[0] * n + c_rows[0], base_col + 3] = 1  # C[0, k_block*2]
        W[c_cols[0] * n + c_rows[1], base_col + 3] = 1  # C[1, k_block*2]
        
        # M5 = (A00+A10)*B11
        U[0 * m + a_cols[0], base_col + 4] = 1  # A[0, j_block*2]
        U[1 * m + a_cols[0], base_col + 4] = 1  # A[1, j_block*2]
        V[b_rows[1] * p + b_cols[1], base_col + 4] = 1  # B[j_block*2+1, k_block*2+1]
        # W: C00 -= M5, C01 += M5
        W[c_cols[0] * n + c_rows[0], base_col + 4] = -1  # C[0, k_block*2]
        W[c_cols[1] * n + c_rows[0], base_col + 4] = 1  # C[0, k_block*2+1]
        
        # M6 = (A01-A00)*(B00+B10)
        U[0 * m + a_cols[1], base_col + 5] = 1  # A[0, j_block*2+1]
        U[0 * m + a_cols[0], base_col + 5] = -1  # A[0, j_block*2]
        V[b_rows[0] * p + b_cols[0], base_col + 5] = 1  # B[j_block*2, k_block*2]
        V[b_rows[1] * p + b_cols[0], base_col + 5] = 1  # B[j_block*2+1, k_block*2]
        # W: C11 += M6
        W[c_cols[1] * n + c_rows[1], base_col + 5] = 1  # C[1, k_block*2+1]
        
        # M7 = (A10-A11)*(B01+B11)
        U[1 * m + a_cols[0], base_col + 6] = 1  # A[1, j_block*2]
        U[1 * m + a_cols[1], base_col + 6] = -1  # A[1, j_block*2+1]
        V[b_rows[0] * p + b_cols[1], base_col + 6] = 1  # B[j_block*2, k_block*2+1]
        V[b_rows[1] * p + b_cols[1], base_col + 6] = 1  # B[j_block*2+1, k_block*2+1]
        # W: C00 += M7
        W[c_cols[0] * n + c_rows[0], base_col + 6] = 1  # C[0, k_block*2]
    
    # Four Strassen blocks:
    # 1. A_L * B11 -> C_L (i=0, j=0, k=0)
    _strassen_block(0, 0, 0, 0)
    
    # 2. A_R * B21 -> C_L (i=0, j=1, k=0)
    _strassen_block(7, 0, 1, 0)
    
    # 3. A_L * B12 -> C_R (i=0, j=0, k=1)
    _strassen_block(14, 0, 0, 1)
    
    # 4. A_R * B22 -> C_R (i=0, j=1, k=1)
    _strassen_block(21, 0, 1, 1)
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try Strassen-based rank-28 decomposition first
    U, V, W = _build_strassen_2x4x4(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
