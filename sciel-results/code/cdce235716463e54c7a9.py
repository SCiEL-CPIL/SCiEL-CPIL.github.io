import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses Strassen 2x2x2 rank-7 decomposition via Kronecker product
    to get 4x4x4 rank-49, then two copies for 4x4x8 rank-98.
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    # Standard Strassen: C = A*B for 2x2 matrices
    # U extracts linear combinations of A, V of B, W reconstructs C
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Verify 2x2x2 Strassen: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For 2x2: i,j,k in {0,1}
    U2_flat = U2.reshape(4, 7)  # index by i*2+j
    V2_flat = V2.reshape(4, 7)  # index by j*2+k
    W2_flat = W2.reshape(4, 7)  # index by k*2+i
    
    # Check: sum_r U2_flat[i*2+j, r] * V2_flat[j*2+k, r] * W2_flat[k*2+i, r] == 1 if i==k else 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1.0 if i == k else 0.0
                actual = 0.0
                for r in range(7):
                    actual += U2_flat[i*2+j, r] * V2_flat[j*2+k, r] * W2_flat[k*2+i, r]
                if abs(actual - expected) > 1e-10:
                    # Fallback to naive
                    return construct_naive(n, m, p)
    
    if time.time() > deadline - 50:
        return construct_rank98(n, m, p)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s] where i=2*i1+i2, j=2*j1+j2
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for i in range(4):
            i1, i2 = i // 2, i % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Verify 4x4x4 decomposition with random test
    if time.time() > deadline - 30:
        return construct_rank98(n, m, p)
    
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((4, 4))
    B_test = rng.standard_normal((4, 4))
    C_expected = A_test @ B_test
    
    # Reconstruct C from decomposition
    U4_np = U4
    V4_np = V4
    W4_np = W4
    C_recon = np.zeros((4, 4))
    for r in range(49):
        u_vec = U4_np[:, r]  # shape (16,)
        v_vec = V4_np[:, r]  # shape (16,)
        w_vec = W4_np[:, r]  # shape (16,)
        # u_vec indexed by i*4+j, v_vec by j*4+k, w_vec by k*4+i
        # C[i,k] += sum_j u_vec[i*4+j] * v_vec[j*4+k] * w_vec[k*4+i]
        for i in range(4):
            for k in range(4):
                s = 0.0
                for j in range(4):
                    s += u_vec[i*4+j] * v_vec[j*4+k] * w_vec[k*4+i]
                C_recon[i, k] += s
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        return construct_naive(n, m, p)
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3): B[:, 0:4]
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7): B[:, 4:8]
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
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
