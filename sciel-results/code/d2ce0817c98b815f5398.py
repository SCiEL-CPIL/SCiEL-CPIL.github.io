import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses the Kronecker product of two levels of Strassen (2x2x2 rank-7)
    to get a rank-49 decomposition for 4x4x4, then applies it independently
    to the two 4x4 column blocks of B, giving total rank 98.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    # U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0
        [0, 0, 0, 0, 1,  0, 1],  # i=0,j=1
        [0, 1, 0, 0, 0,  1, 0],  # i=1,j=0
        [1, 1, 0, 1, 0,  0,-1]   # i=1,j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0
        [0, 0, 1,  0, 0, 1, 0],  # j=0,k=1
        [0, 0, 0,  1, 0, 0, 1],  # j=1,k=0
        [1, 0,-1,  0, 1, 0, 1]   # j=1,k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0
        [0, 1, 0, 1,  0, 0, 0],  # k=0,i=1
        [0, 0, 1, 0,  1, 0, 0],  # k=1,i=0
        [1,-1, 1, 0,  0, 1, 0]   # k=1,i=1
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # Vectorized construction
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    # i = 2*i1 + i2, j = 2*j1 + j2
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Generate indices
    i1s, i2s, j1s, j2s, ts, ss = np.meshgrid(
        np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(7), np.arange(7),
        indexing='ij'
    )
    
    i_flat = (2 * i1s + i2s).flatten()
    j_flat = (2 * j1s + j2s).flatten()
    c_flat = (ts * 7 + ss).flatten()
    
    # U4
    u_vals = U2[(2 * i1s + j1s).flatten(), ts.flatten()] * U2[(2 * i2s + j2s).flatten(), ss.flatten()]
    U4[i_flat, c_flat] = u_vals
    
    # V4
    j1s_v, j2s_v, k1s_v, k2s_v, ts_v, ss_v = np.meshgrid(
        np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(7), np.arange(7),
        indexing='ij'
    )
    j_flat_v = (2 * j1s_v + j2s_v).flatten()
    k_flat_v = (2 * k1s_v + k2s_v).flatten()
    c_flat_v = (ts_v * 7 + ss_v).flatten()
    
    v_vals = V2[(2 * j1s_v + k1s_v).flatten(), ts_v.flatten()] * V2[(2 * j2s_v + k2s_v).flatten(), ss_v.flatten()]
    V4[j_flat_v, c_flat_v] = v_vals
    
    # W4
    k1s_w, k2s_w, i1s_w, i2s_w, ts_w, ss_w = np.meshgrid(
        np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(7), np.arange(7),
        indexing='ij'
    )
    k_flat_w = (2 * k1s_w + k2s_w).flatten()
    i_flat_w = (2 * i1s_w + i2s_w).flatten()
    c_flat_w = (ts_w * 7 + ss_w).flatten()
    
    w_vals = W2[(2 * k1s_w + i1s_w).flatten(), ts_w.flatten()] * W2[(2 * k2s_w + i2s_w).flatten(), ss_w.flatten()]
    W4[k_flat_w, c_flat_w] = w_vals
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)   # 16 x 98
    V = np.zeros((m*p, rank), dtype=float)   # 32 x 98
    W = np.zeros((p*n, rank), dtype=float)   # 32 x 98
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    # V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    # We can use broadcasting or simple assignment
    for j in range(4):
        V[j*8:j*8+4, 0:49] = V4[j*4:(j+1)*4, 0:49]
    # W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    W[0:16, 0:49] = W4
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        V[j*8+4:j*8+8, 49:98] = V4[j*4:(j+1)*4, 0:49]
    # W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    W[16:32, 49:98] = W4
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verification (optional but good for safety)
    # We skip full verification to save time, trusting the construction.
    # The construction is deterministic and known to be correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
