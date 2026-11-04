import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 8
    
    # Construct rank 98 decomposition
    result = construct_rank98(n, m, p)
    
    # Verify correctness
    if time.time() < deadline - 5:
        _verify_decomposition(result, n, m, p)
    
    return result

def construct_rank98(n, m, p):
    # 2x2 Strassen decomposition (rank 7)
    # Indices: U[i, t], V[j, t], W[k, t]
    # i, j, k in {0, 1}
    # t in {0, ..., 6}
    
    # Standard Strassen matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_S = np.array([
        [1, 1, 1, 0, 1, -1, 0],  # A00
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=float)
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 1], # B00
        [1, 0, 1, 1, 1, 0, 1]   # B11
    ], dtype=float)
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 1, 0, 0, -1, 0]  # C11
    ], dtype=float)
    
    # Check 2x2x2 decomposition
    # T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]
    # Should be 1 for all i,j,k in {0,1}
    
    # Build 4x4x4 decomposition (rank 49) using Kronecker product
    # U4[i1*2+i2, t1*7+t2] = U_S[i1, t1] * U_S[i2, t2]
    # V4[j1*2+j2, t1*7+t2] = V_S[j1, t1] * V_S[j2, t2]
    # W4[k1*2+k2, t1*7+t2] = W_S[k1, t1] * W_S[k2, t2]
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for t1 in range(7):
                for t2 in range(7):
                    idx_i = i1*2 + i2
                    idx_t = t1*7 + t2
                    U4[idx_i, idx_t] = U_S[i1, t1] * U_S[i2, t2]
                    
    for j1 in range(2):
        for j2 in range(2):
            for t1 in range(7):
                for t2 in range(7):
                    idx_j = j1*2 + j2
                    idx_t = t1*7 + t2
                    V4[idx_j, idx_t] = V_S[j1, t1] * V_S[j2, t2]
                    
    for k1 in range(2):
        for k2 in range(2):
            for t1 in range(7):
                for t2 in range(7):
                    idx_k = k1*2 + k2
                    idx_t = t1*7 + t2
                    W4[idx_k, idx_t] = W_S[k1, t1] * W_S[k2, t2]
    
    # Extend to 4x4x8 (rank 98)
    # B is 4x8, so we have two 4x4 blocks B1, B2
    # C is 4x8, so two 4x4 blocks C1, C2
    # C1 = A B1, C2 = A B2
    # Use the same U4 for both
    # V4 for B1, V4 for B2
    # W4 for C1, W4 for C2
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float) # 4*8
    W = np.zeros((32, rank), dtype=float) # 8*4
    
    # First 49 factors for B1, C1
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, 0:49]
            
    # Next 49 factors for B2, C2
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Verify tensor identity: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                max_err = max(max_err, abs(val - 1.0))
                
    if max_err > 1e-9:
        raise ValueError(f"Decomposition error: {max_err}")
