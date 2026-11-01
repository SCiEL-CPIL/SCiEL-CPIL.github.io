import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, num_tests=10):
    """Check that the decomposition is exact for random inputs using NumPy for speed."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for seed in range(num_tests):
        random.seed(seed)
        A = np.array([[random.randint(-3, 3) for _ in range(m)] for _ in range(n)], dtype=float)
        B = np.array([[random.randint(-3, 3) for _ in range(p)] for _ in range(m)], dtype=float)
        
        # Compute AB directly
        AB = A @ B
        
        # Compute using decomposition
        # X[r, i] = sum_j U[i*m+j, r] * A[i, j]
        # U is (n*m, rank). Reshape to (n, m, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        # A is (n, m). We want to sum over m.
        # X[r, i] = sum_j U_reshaped[i, j, r] * A[i, j]
        # This is a batched dot product? 
        # X[r, i] = <U_reshaped[i, :, r], A[i, :]>
        # Let's do it with einsum: 'ijr,ij->ri'
        X = np.einsum('ijr,ij->ri', U_reshaped, A)
        
        # Y[r, k] = sum_j V[j*p+k, r] * B[j, k]
        # V is (m*p, rank). Reshape to (m, p, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        # B is (m, p).
        # Y[r, k] = sum_j V_reshaped[j, k, r] * B[j, k]
        # 'jkr,jk->rk'
        Y = np.einsum('jkr,jk->rk', V_reshaped, B)
        
        # C[i, k] = sum_r W[k*n+i, r] * X[r, i] * Y[r, k]
        # W is (p*n, rank). Reshape to (p, n, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        # We need C[i, k].
        # W[k, i, r] * X[r, i] * Y[r, k]
        # 'kir,ri,rk->ik'
        C = np.einsum('kir,ri,rk->ik', W_reshaped, X, Y)
        
        if not np.allclose(C, AB, atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x6():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # U_s[i*m+j][r]
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V_s[j*p+k][r] for local 2x2
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 1, 0],   # B01
        [0, 0, 1, 0, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W_s[k*n+i][r] for local 2x2
    W_s = [
        [1, 0, 0, -1, 1, -1, 1], # C00
        [0, 0, 1, 1, 0, 0, 1],   # C01
        [1, 1, 0, 0, -1, 0, -1], # C10
        [0, 1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Apply to 3 blocks of 2 columns
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            
            # U is shared across blocks in terms of A, but each block has its own rank index
            for ij in range(4):
                U[ij][r] = U_s[ij][s]
            
            # V maps local j,k to global j, k_global
            for j in range(2):
                for k_local in range(2):
                    k_global = block * 2 + k_local
                    v_idx = j * 6 + k_global
                    v_s_idx = j * 2 + k_local
                    V[v_idx][r] = V_s[v_s_idx][s]
            
            # W maps local k,i to global k, i
            for k_local in range(2):
                for i in range(2):
                    k_global = block * 2 + k_local
                    w_idx = k_global * 2 + i
                    w_s_idx = k_local * 2 + i
                    W[w_idx][r] = W_s[w_s_idx][s]
                    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    
    # Try Strassen
    result = _strassen_2x2x6()
    
    # Check validity
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if Strassen fails (should not happen)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
