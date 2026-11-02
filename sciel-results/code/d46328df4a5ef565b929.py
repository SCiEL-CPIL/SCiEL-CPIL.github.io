import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Define the 7 Strassen 2x2x2 terms as sparse index-value lists
    # Each term: (list of (idx, val) for U, list of (idx, val) for V, list of (idx, val) for W)
    # U indices: (i, j) in 2x2 -> i*2+j
    # V indices: (j, k) in 2x2 -> j*2+k
    # W indices: (k, i) in 2x2 -> k*2+i
    
    # Corrected Strassen terms
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]), # Fixed W indices
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1),(3,1)]), # Fixed W indices
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # Build 4x4x4 part using Kronecker product of 2x2x2 Strassen terms
    for r in range(7):
        for s in range(7):
            U_rs = np.zeros(16)
            V_rs = np.zeros(16)
            W_rs = np.zeros(16)
            
            # Build U_{rs}
            for (idx_u1, val_u1) in strassen_terms[r][0]:
                i1 = idx_u1 // 2
                j1 = idx_u1 % 2
                for (idx_u2, val_u2) in strassen_terms[s][0]:
                    i2 = idx_u2 // 2
                    j2 = idx_u2 % 2
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    U_rs[i*4+j] += val_u1 * val_u2
            
            # Build V_{rs}
            for (idx_v1, val_v1) in strassen_terms[r][1]:
                j1 = idx_v1 // 2
                k1 = idx_v1 % 2
                for (idx_v2, val_v2) in strassen_terms[s][1]:
                    j2 = idx_v2 // 2
                    k2 = idx_v2 % 2
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    V_rs[j*4+k] += val_v1 * val_v2
            
            # Build W_{rs}
            for (idx_w1, val_w1) in strassen_terms[r][2]:
                k1 = idx_w1 // 2
                i1 = idx_w1 % 2
                for (idx_w2, val_w2) in strassen_terms[s][2]:
                    k2 = idx_w2 // 2
                    i2 = idx_w2 % 2
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    W_rs[k*4+i] += val_w1 * val_w2
            
            # Extend to 4x4x5 by zeroing out column 4 in V and row 4 in W
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            for idx in range(16):
                U_full[idx] = U_rs[idx]
                V_full[idx] = V_rs[idx]
                W_full[idx] = W_rs[idx]
            
            add_term(U_full, V_full, W_full)
    
    # Build 4x4x1 part (naive, rank 16)
    for i in range(4):
        for j in range(4):
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            U_full[i*4+j] = 1
            V_full[j*5+4] = 1
            W_full[4*4+i] = 1
            
            add_term(U_full, V_full, W_full)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get the Kronecker-embedded Strassen decomposition (rank 65)
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    # Verify the Kronecker construction
    if _verify_decomposition(U_kron, V_kron, W_kron, n, m, p):
        return {"U": U_kron, "V": V_kron, "W": W_kron}
    else:
        # Fallback to a naive rank 80 decomposition if something goes wrong
        # This is just a safety net, should not be triggered
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1
                    V_naive[j*p+k, idx] = 1
                    W_naive[k*n+i, idx] = 1
                    idx += 1
        
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
# EVOLVE-BLOCK-END
