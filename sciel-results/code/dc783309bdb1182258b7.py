import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Generate target tensor
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def add_schoolblock(I, J, K):
        for i in I:
            for j in J:
                for k in K:
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    u = np.zeros(n * m); u[a] = 1.0
                    v = np.zeros(m * p); v[b] = 1.0
                    w = np.zeros(n * p); w[c] = 1.0
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    def add_strassen_block(I, J, K):
        # Standard Strassen for 2x2x2
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        a11 = i1 * m + j1; a12 = i1 * m + j2
        a21 = i2 * m + j1; a22 = i2 * m + j2
        b11 = j1 * p + k1; b12 = j1 * p + k2
        b21 = j2 * p + k1; b22 = j2 * p + k2
        c11 = k1 * n + i1; c12 = k2 * n + i1
        c21 = k1 * n + i2; c22 = k2 * n + i2
        
        # 7 terms
        # P1
        u = np.zeros(n*m); u[a11] = 1
        v = np.zeros(m*p); v[b11] = 1; v[b21] = -1
        w = np.zeros(n*p); w[c11] = 1; w[c22] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P2
        u = np.zeros(n*m); u[a11] = 1; u[a21] = 1
        v = np.zeros(m*p); v[b21] = 1
        w = np.zeros(n*p); w[c11] = -1; w[c21] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P3
        u = np.zeros(n*m); u[a12] = 1; u[a22] = -1
        v = np.zeros(m*p); v[b22] = 1
        w = np.zeros(n*p); w[c12] = 1; w[c22] = -1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P4
        u = np.zeros(n*m); u[a22] = 1
        v = np.zeros(m*p); v[b12] = 1; v[b11] = -1
        w = np.zeros(n*p); w[c11] = 1; w[c21] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P5
        u = np.zeros(n*m); u[a11] = 1; u[a12] = 1
        v = np.zeros(m*p); v[b11] = 1
        w = np.zeros(n*p); w[c11] = 1; w[c12] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P6
        u = np.zeros(n*m); u[a21] = 1; u[a11] = -1
        v = np.zeros(m*p); v[b11] = 1; v[b12] = 1
        w = np.zeros(n*p); w[c12] = 1; w[c21] = 1 # Guessing C formula for P6?
        # Actually, let's just add it and let the verifier check.
        # If the verifier fails, we discard this block.
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # P7
        u = np.zeros(n*m); u[a12] = 1; u[a22] = -1
        v = np.zeros(m*p); v[b21] = 1; v[b22] = 1
        w = np.zeros(n*p); w[c22] = 1 # Guessing
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # Try to build with Strassen
    I_parts = [[0, 1], [2, 3]]
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4]]
    
    U_strassen = []
    V_strassen = []
    W_strassen = []
    
    # Save current state
    U_init = len(U_cols)
    
    for I in I_parts:
        for J in J_parts:
            for K in K_parts:
                if len(I) == 2 and len(J) == 2 and len(K) == 2:
                    add_strassen_block(I, J, K)
                else:
                    add_schoolblock(I, J, K)
                    
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    # Verify
    # S = sum(U_r V_r W_r)
    # This is a tensor contraction.
    # S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Can be done as (U.T @ V) * W? No.
    # S = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    S = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    if not np.allclose(S, T):
        # Revert to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    u = np.zeros(n * m); u[a] = 1.0
                    v = np.zeros(m * p); v[b] = 1.0
                    w = np.zeros(n * p); w[c] = 1.0
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    return {"U": np.column_stack(U_cols).tolist(), "V": np.column_stack(V_cols).tolist(), "W": np.column_stack(W_cols).tolist()}
