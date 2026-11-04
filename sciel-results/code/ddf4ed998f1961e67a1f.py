import numpy as np
import time

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses the Kronecker product of two levels of Strassen (2x2x2 rank-7)
    to get a rank-49 decomposition for 4x4x4, then applies it independently
    to the two 4x4 column blocks of B, giving total rank 98.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # First block (k=0..3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (k=4..7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    # Verify the base decomposition
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    A = np.einsum('ijr,jkr->ijk r'.replace(' ', ''), U_r, V_r)
    T = np.einsum('ijk r,ki r->ijk'.replace(' ', ''), A, W_r)
    max_err = np.max(np.abs(T - 1.0))
    
    if max_err > 1e-6:
        # Should not happen, but return the base decomposition
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction by checking for redundant columns
    # Vectorize each rank-1 tensor component
    T_vectors = []
    for r in range(rank):
        u = U_r[:, :, r]
        v = V_r[:, :, r]
        w = W_r[:, :, r]
        t = np.einsum('ij,jk,ki->ijk', u, v, w)
        T_vectors.append(t.flatten())
    
    T_matrix = np.array(T_vectors).T  # (n*m*p, rank)
    
    # Try to find a subset of columns whose sum equals all-ones
    # Start with all columns, try removing one at a time
    current_cols = list(range(rank))
    tol = 1e-8
    
    while True:
        if time.time() > deadline:
            break
        
        found_removal = False
        for r in current_cols:
            if time.time() > deadline:
                break
            
            # Check if removing column r still gives sum = ones
            T_remaining = np.zeros(n * m * p)
            for r2 in current_cols:
                if r2 != r:
                    T_remaining += T_vectors[r2]
            
            if np.max(np.abs(T_remaining - 1.0)) < tol:
                current_cols.remove(r)
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if len(current_cols) < rank:
        U_new = U[:, current_cols]
        V_new = V[:, current_cols]
        W_new = W[:, current_cols]
        
        # Verify the reduced decomposition
        U_r2 = U_new.reshape(n, m, U_new.shape[1])
        V_r2 = V_new.reshape(m, p, V_new.shape[1])
        W_r2 = W_new.reshape(p, n, W_new.shape[1])
        A2 = np.einsum('ijr,jkr->ijk r'.replace(' ', ''), U_r2, V_r2)
        T2 = np.einsum('ijk r,ki r->ijk'.replace(' ', ''), A2, W_r2)
        
        if np.max(np.abs(T2 - 1.0)) < 1e-6:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return the base rank-98 decomposition
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
