"""Matrix multiplication 3,4,5 decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T with shape (n*m, m*p, p*n).
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the decomposition U, V, W exactly reproduces the target tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the decomposed tensor D
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    return np.allclose(D, T, atol=tol, rtol=0)

def _get_strassen_2x2x2():
    """
    Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen.
    u, v, w are flat vectors of length 4 representing 2x2 matrices.
    Indexing: [00, 01, 10, 11]
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # C00 += M1, C11 += M1
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))

    # M2 = (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    # C10 += M2, C11 -= M2
    w2 = [0, 0, 1, -1]
    terms.append((u2, v2, w2))

    # M3 = A00 B11
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 0, 1]
    # C01 += M3, C11 += M3
    w3 = [0, 1, 0, 1]
    terms.append((u3, v3, w3))

    # M4 = A11 B10
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, 0, 0]
    # C00 -= M4, C10 += M4
    w4 = [-1, 0, 1, 0]
    terms.append((u4, v4, w4))

    # M5 = (A00+A01)B10
    u5 = [1, 1, 0, 0]
    v5 = [0, 1, 0, 0]
    # C00 += M5
    w5 = [1, 0, 0, 0]
    terms.append((u5, v5, w5))

    # M6 = (A10-A00)(B00+B11)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 0, 0, 1]
    # C11 -= M6
    w6 = [0, 0, 0, -1]
    terms.append((u6, v6, w6))

    # M7 = (A01-A11)(B10+B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    # C01 += M7, C11 -= M7
    w7 = [0, 1, 0, -1]
    terms.append((u7, v7, w7))
    
    return terms

def _construct_rank_59(n, m, p):
    """
    Construct a rank 59 decomposition by using Strassen for a 2x2x2 sub-block
    and schoolbook for the rest.
    """
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 2x2x2 block indices
    # A rows: 0, 1; A cols: 0, 1
    # B rows: 0, 1; B cols: 0, 1
    # C rows: 0, 1; C cols: 0, 1
    
    a_rows = [0, 1]
    a_cols = [0, 1]
    b_rows = [0, 1]
    b_cols = [0, 1]
    c_rows = [0, 1]
    c_cols = [0, 1]
    
    # Map local 2x2 indices to global indices
    # Local index: r*2 + c
    # Global U index: a_row * m + a_col
    # Global V index: b_row * p + b_col
    # Global W index: c_col * n + c_row
    
    def map_u(local_r, local_c):
        return a_rows[local_r] * m + a_cols[local_c]
        
    def map_v(local_r, local_c):
        return b_rows[local_r] * p + b_cols[local_c]
        
    def map_w(local_r, local_c):
        # W index is k*n + i, where k is col of C, i is row of C
        # Local C is 2x2. Local r is row, local c is col.
        return c_cols[local_c] * n + c_rows[local_r]

    # 1. Add Strassen terms (7 terms)
    strassen_terms = _get_strassen_2x2x2()
    for r_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
        for lr in range(2):
            for lc in range(2):
                u_val = u_loc[lr * 2 + lc]
                if u_val != 0:
                    U[map_u(lr, lc), r_idx] = u_val
                    
                v_val = v_loc[lr * 2 + lc]
                if v_val != 0:
                    V[map_v(lr, lc), r_idx] = v_val
                    
                w_val = w_loc[lr * 2 + lc]
                if w_val != 0:
                    W[map_w(lr, lc), r_idx] = w_val

    # 2. Add Schoolbook terms for the rest
    # The full set of indices is (i, j, k) for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # The Strassen block covers i in {0,1}, j in {0,1}, k in {0,1}
    
    col = 7 # Start after Strassen terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this index is part of the Strassen block
                if i in a_rows and j in a_cols and k in b_cols:
                    continue
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Check if we used exactly 59 columns
    if col != rank:
        # If the count is wrong, adjust rank or raise error
        # For 3x4x5, 60 - 8 + 7 = 59. It should be correct.
        pass
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Attempt Rank 59 construction
    try:
        U, V, W = _construct_rank_59(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            print("Found valid rank 59 decomposition", file=sys.stderr)
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            print("Rank 59 construction failed verification", file=sys.stderr)
    except Exception as e:
        print(f"Error in rank 59 construction: {e}", file=sys.stderr)

    # 2. Fallback to Schoolbook (Rank 60)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
