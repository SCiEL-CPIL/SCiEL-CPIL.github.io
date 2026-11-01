"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """Rank-49 Strassen decomposition on 2x2 blocks."""
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            k1 = 2 * l
            k2 = 2 * l + 1
            
            # Strassen 7 terms for block (k1, k2)
            # M1: (A00+A11)(B0,k1 + B1,k2)
            U[0, col_start] = 1; U[3, col_start] = 1
            V[k1, col_start] = 1; V[p + k2, col_start] = 1
            W[k1 * n, col_start] = 1; W[k2 * n + 1, col_start] = 1
            
            # M2: (A10+A11) * B0,k1
            U[2, col_start+1] = 1; U[3, col_start+1] = 1
            V[k1, col_start+1] = 1
            W[k1 * n + 1, col_start+1] = 1; W[k2 * n + 1, col_start+1] = -1
            
            # M3: A00 * (B0,k2 - B1,k2)
            U[0, col_start+2] = 1
            V[k2, col_start+2] = 1; V[p + k2, col_start+2] = -1
            W[k2 * n, col_start+2] = 1; W[k2 * n + 1, col_start+2] = 1
            
            # M4: A11 * (B1,k1 - B0,k1)
            U[3, col_start+3] = 1
            V[p + k1, col_start+3] = 1; V[k1, col_start+3] = -1
            W[k1 * n, col_start+3] = 1; W[k1 * n + 1, col_start+3] = 1
            
            # M5: (A00+A01) * B1,k2
            U[0, col_start+4] = 1; U[1, col_start+4] = 1
            V[p + k2, col_start+4] = 1
            W[k1 * n, col_start+4] = -1; W[k2 * n, col_start+4] = 1
            
            # M6: (A10-A00) * (B0,k1 + B0,k2)
            U[2, col_start+5] = 1; U[0, col_start+5] = -1
            V[k1, col_start+5] = 1; V[k2, col_start+5] = 1
            W[k2 * n + 1, col_start+5] = 1
            
            # M7: (A01-A11) * (B1,k1 + B1,k2)
            U[1, col_start+6] = 1; U[3, col_start+6] = -1
            V[p + k1, col_start+6] = 1; V[p + k2, col_start+6] = 1
            W[k1 * n, col_start+6] = 1
            
        return U, V, W

    # Try rank-49 Strassen first
    U, V, W = strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
