import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Rows correspond to indices i*2+j for i,j in {0,1}
    
    # Initialize 2x2 Strassen factors
    # U2[i*2+j, k] = coeff for A_ij in M_k
    # V2[j*2+k, k] = coeff for B_jk in M_k
    # W2[k*2+i, k] = coeff for C_ki in M_k
    
    U2 = np.zeros((4, 7), dtype=np.int8)
    V2 = np.zeros((4, 7), dtype=np.int8)
    W2 = np.zeros((4, 7), dtype=np.int8)
    
    # M0 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M2 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M3 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M4 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M6 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4 factors using Kronecker product structure
    # Rank is 7 * 7 = 49
    R = 49
    U = np.zeros((n*m, R), dtype=np.int8)
    V = np.zeros((m*p, R), dtype=np.int8)
    W = np.zeros((p*n, R), dtype=np.int8)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Pre-compute outer products for U, V, W
    # This avoids repeated np.outer calls in the loop if we were to search,
    # but here we just fill the arrays.
    
    # U construction
    for i in range(n):
        for j in range(m):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            idx_u1 = i1 * 2 + j1
            idx_u2 = i2 * 2 + j2
            U[i*m + j] = np.outer(U2[idx_u1, :], U2[idx_u2, :]).flatten()
            
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # V construction
    for j in range(m):
        for k in range(p):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            idx_v1 = j1 * 2 + k1
            idx_v2 = j2 * 2 + k2
            V[j*p + k] = np.outer(V2[idx_v1, :], V2[idx_v2, :]).flatten()
            
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # W construction
    for k in range(p):
        for i in range(n):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            idx_w1 = k1 * 2 + i1
            idx_w2 = k2 * 2 + i2
            W[k*n + i] = np.outer(W2[idx_w1, :], W2[idx_w2, :]).flatten()
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
