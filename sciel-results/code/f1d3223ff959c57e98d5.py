import numpy as np
import time

# Pre-compute the Strassen decomposition matrices for 4x4
# Rank = 49
# U, V, W shapes: (16, 49)

def _build_matrices():
    # Strassen decomposition for 2x2 matrices
    # U2: shape (4, 7), index [i*2+j, t]
    # V2: shape (4, 7), index [j*2+k, t]
    # W2: shape (4, 7), index [k*2+i, t]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ], dtype=int)

    # Outer Strassen coefficients
    # beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: M0 = (A00+A11)(B00+B11)
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1
    
    # s=1: M1 = (A10+A11)B00
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1
    
    # s=2: M2 = A00(B01-B11)
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1
    
    # s=3: M3 = A11(B10-B00)
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1
    
    # s=4: M4 = (A00+A01)B11
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1
    
    # s=5: M5 = (A10-A00)(B00+B01)
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1
    
    # s=6: M6 = (A01-A11)(B10+B11)
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Construct U, V, W of shape (16, 49)
    # U[r*4+c, s*7+t] = sum_{I,J} beta[s,I,J] * U2[(2*I+i2)*2 + (2*J+j2), t] 
    # This is effectively a Kronecker-like product structure.
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Vectorized construction
    # We can view U as a tensor U[r, c, s, t] where r=2I+i2, c=2J+j2
    # U[r, c, s, t] = sum_{I,J} beta[s, I, J] * U2[2I+i2, 2J+j2, t]? No.
    # U2 is indexed by [i2*2+j2, t].
    # Let's stick to loops but make them tight, or precompute.
    # Given the small size (16x49), loops are fine if we don't use np.sum in a loop.
    
    for I in range(2):
        for J in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    row_u = (2*I + i2)*4 + (2*J + j2)
                    u2_idx = i2*2 + j2
                    # beta[:, I, J] is shape (7,)
                    # U2[u2_idx, :] is shape (7,)
                    # Outer product is (7, 7) -> flatten to (49,)
                    # U[row_u, :] += np.outer(beta[:, I, J], U2[u2_idx, :]).flatten()
                    # Using multiplication and reshape is faster than outer?
                    U[row_u, :] += (beta[:, I, J, None] * U2[u2_idx, None, :]).reshape(49)
                    
    for J in range(2):
        for K in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    row_v = (2*J + j2)*4 + (2*K + k2)
                    v2_idx = j2*2 + k2
                    V[row_v, :] += (gamma[:, J, K, None] * V2[v2_idx, None, :]).reshape(49)
                    
    for I in range(2):
        for K in range(2):
            for i2 in range(2):
                for k2 in range(2):
                    row_w = (2*K + k2)*4 + (2*I + i2)
                    w2_idx = k2*2 + i2
                    W[row_w, :] += (delta[:, I, K, None] * W2[w2_idx, None, :]).reshape(49)
                    
    return U, V, W

_U, _V, _W = _build_matrices()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Return the pre-computed matrices
    # Convert to list of lists as required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The example returns .tolist()
    
    return {"U": _U.tolist(), "V": _V.tolist(), "W": _W.tolist()}
