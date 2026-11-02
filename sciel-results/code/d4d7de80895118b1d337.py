import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    # Build the target tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a rank-10 decomposition for the 2x3x3x2 block
    T_block = np.zeros((6, 6, 4), dtype=np.int64)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_block[i*3+j, j*2+k, k*2+i] = 1
    
    # Try rank 10 using gradient descent
    np.random.seed(123)
    rank = 10
    U = np.random.randn(6, rank) * 0.5
    V = np.random.randn(6, rank) * 0.5
    W = np.random.randn(4, rank) * 0.5
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for iteration in range(20000):
        if time.time() > deadline - 20:
            break
        
        T_curr = np.einsum('ad,bd,cd->abc', U, V, W)
        err = np.sum((T_curr - T_block)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-6:
            break
        
        lr = 0.01
        dU = np.einsum('abc,bd,cd->ad', T_curr - T_block, V, W)
        dV = np.einsum('abc,ad,cd->bd', T_curr - T_block, U, W)
        dW = np.einsum('abc,ad,bd->cd', T_curr - T_block, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    U_r = np.round(best_U)
    V_r = np.round(best_V)
    W_r = np.round(best_W)
    
    T_check = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
    if np.array_equal(T_check, T_block):
        U40 = np.zeros((n*m, 40), dtype=np.int64)
        V40 = np.zeros((m*p, 40), dtype=np.int64)
        W40 = np.zeros((p*n, 40), dtype=np.int64)
        
        col_offset = 0
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            for d in range(rank):
                for a_local in range(6):
                    if U_r[a_local, d] != 0:
                        i = a_local // 3
                        j = a_local % 3
                        a_global = i * m + j
                        U40[a_global, col_offset + d] += U_r[a_local, d]
                
                for b_local in range(6):
                    if V_r[b_local, d] != 0:
                        j = b_local // 2
                        k = b_local % 2
                        k_global = k0 + k
                        b_global = j * p + k_global
                        V40[b_global, col_offset + d] += V_r[b_local, d]
                
                for c_local in range(4):
                    if W_r[c_local, d] != 0:
                        k = c_local // 2
                        i = c_local % 2
                        k_global = k0 + k
                        c_global = k_global * n + i
                        W40[c_global, col_offset + d] += W_r[c_local, d]
            
            col_offset += rank
        
        def verify(U, V, W):
            r = U.shape[1]
            T_check = np.einsum('ad,bd,cd->abc', U, V, W)
            return np.array_equal(T_check, T_exp)
        
        if verify(U40, V40, W40):
            return {"U": U40.tolist(), "V": V40.tolist(), "W": W40.tolist()}
    
    # Fallback to rank 44
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
    col_offset = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = -1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 2, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        V[1*p + k0, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        U[0*m + 2, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 2, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
