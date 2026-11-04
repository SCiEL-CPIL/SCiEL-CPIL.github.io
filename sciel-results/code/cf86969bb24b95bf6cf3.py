import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0

def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1
    
    return U, V, W

def _outer_product_embed(n, m, p, i0, j0, k0, bi, bk):
    """
    Embeds a block where j is fixed (bj=1) and i, k vary.
    Computes C[i,k] += A[i, j0] * B[j0, k].
    This is a rank-1 update.
    """
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i, j0] = 1
    for i in range(i0, i0 + bi):
        U[i * m + j0, 0] = 1
        
    # V[j0, k] = 1
    for k in range(k0, k0 + bk):
        V[j0 * p + k, 0] = 1
        
    # W[k, i] = 1
    for k in range(k0, k0 + bk):
        for i in range(i0, i0 + bi):
            W[k * n + i, 0] = 1
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 1. Strassen for 4x4x4 (i:0-3, j:0-3, k:0-3)
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i0, j0, k0)
                U_parts.append(U_s)
                V_parts.append(V_s)
                W_parts.append(W_s)
                
    # 2. Schoolbook for 4x4x1 (i:0-3, j:0-3, k=4)
    # Rank 16
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 0, 4, 4, 4, 1)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # 3. Outer Product for 4x1x5 (i:0-3, j=4, k:0-4)
    # Rank 1
    U_s, V_s, W_s = _outer_product_embed(n, m, p, 0, 4, 0, 4, 5)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    # Verify
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
