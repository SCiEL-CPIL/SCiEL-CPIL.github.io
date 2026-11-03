"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Rank-55 decomposition: 20 (j=0) + 5*7 (Strassen on 2x2 blocks for j=1,2)
    rank = 55
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part A: j=0, schoolbook for 2x1x10
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part B: j=1,2 with 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1
        W[w_01, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1
    
    # Verify the decomposition
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    T_reconstructed = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for c in range(rank):
        T_reconstructed += np.einsum('i,j,k->ijk', U[:, c].astype(float), V[:, c].astype(float), W[:, c].astype(float))
    
    T_true = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1
    
    if not np.allclose(T_reconstructed, T_true, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=int)
        V_school = np.zeros((m * p, rank_school), dtype=int)
        W_school = np.zeros((p * n, rank_school), dtype=int)
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col_s] = 1
                    V_school[j * p + k, col_s] = 1
                    W_school[k * n + i, col_s] = 1
                    col_s += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to reduce rank by searching for a rank-54 decomposition
    # This is a complex optimization problem, but we can try a simple approach
    # Check if any column can be expressed as a linear combination of others
    # For now, return the valid rank-55 decomposition
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
