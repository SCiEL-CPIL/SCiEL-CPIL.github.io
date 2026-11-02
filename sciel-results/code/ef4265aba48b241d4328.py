import numpy as np
import time

def construct():
    start_time = time.time()
    # n=2, m=3, p=5
    n, m, p = 2, 3, 5
    
    # Total rank 28
    R = 28
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    # Block 1: j=0 (2x1x5)
    # U index: i*m + 0
    # V index: 0*p + k = k
    # W index: k*n + i
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Block 2: j=1,2 (2x2x5)
    # j in {1, 2}
    
    # Sub-block 2a: k=0,1 (2x2x2) Strassen
    # Local indices: i' in {0,1}, j' in {0,1} (global j=1+j'), k' in {0,1} (global k=0+k')
    # U idx: i*m + (1+j')
    # V idx: (1+j')*p + k'
    # W idx: k'*n + i
    
    # Strassen 7 terms
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0, j=1. A11: i=1, j=2.
    # B00: j=1, k=0. B11: j=2, k=1.
    # C00: k=0, i=0. C11: k=1, i=1.
    
    def add_strassen_to_arrays(U, V, W, i_off, j_off, k_off, col_offset, n, m, p):
        c = col_offset
        
        # M1
        U[i_off * m + j_off, c] = 1
        U[(i_off + 1) * m + (j_off + 1), c] = 1
        V[j_off * p + k_off, c] = 1
        V[(j_off + 1) * p + (k_off + 1), c] = 1
        W[k_off * n + i_off, c] = 1
        W[(k_off + 1) * n + (i_off + 1), c] = 1
        c += 1
        
        # M2
        U[(i_off + 1) * m + j_off, c] = 1
        U[(i_off + 1) * m + (j_off + 1), c] = 1
        V[j_off * p + k_off, c] = 1
        W[k_off * n + (i_off + 1), c] = 1
        W[(k_off + 1) * n + (i_off + 1), c] = -1
        c += 1
        
        # M3
        U[i_off * m + j_off, c] = 1
        V[j_off * p + (k_off + 1), c] = 1
        V[(j_off + 1) * p + (k_off + 1), c] = -1
        W[(k_off + 1) * n + i_off, c] = 1
        W[(k_off + 1) * n + (i_off + 1), c] = 1
        c += 1
        
        # M4
        U[(i_off + 1) * m + (j_off + 1), c] = 1
        V[(j_off + 1) * p + k_off, c] = 1
        V[j_off * p + k_off, c] = -1
        W[k_off * n + i_off, c] = 1
        W[k_off * n + (i_off + 1), c] = 1
        c += 1
        
        # M5
        U[i_off * m + j_off, c] = 1
        U[i_off * m + (j_off + 1), c] = 1
        V[(j_off + 1) * p + (k_off + 1), c] = 1
        W[k_off * n + i_off, c] = -1
        W[(k_off + 1) * n + i_off, c] = 1
        c += 1
        
        # M6
        U[(i_off + 1) * m + j_off, c] = 1
        U[i_off * m + j_off, c] = -1
        V[j_off * p + k_off, c] = 1
        V[j_off * p + (k_off + 1), c] = 1
        W[(k_off + 1) * n + (i_off + 1), c] = 1
        c += 1
        
        # M7
        U[i_off * m + (j_off + 1), c] = 1
        U[(i_off + 1) * m + (j_off + 1), c] = -1
        V[(j_off + 1) * p + k_off, c] = 1
        V[(j_off + 1) * p + (k_off + 1), c] = 1
        W[k_off * n + i_off, c] = 1
        c += 1
        
        return c

    # Call for Block 2a (i_off=0, j_off=1, k_off=0)
    col = add_strassen_to_arrays(U, V, W, 0, 1, 0, col, n, m, p)
    
    # Sub-block 2b: k=2,3 (2x2x2) Strassen
    # j in {1,2}, k in {2,3}
    # i_off=0, j_off=1, k_off=2
    col = add_strassen_to_arrays(U, V, W, 0, 1, 2, col, n, m, p)
    
    # Sub-block 2c: k=4 (2x2x1) Schoolbook
    # j in {1,2}, k=4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

    assert col == R
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
