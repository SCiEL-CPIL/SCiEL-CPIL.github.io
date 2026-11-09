import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B
    
    # Decomposition 1: 1x4x3
    n1, m1, p1 = 1, 4, 3
    r1 = m1 * p1
    U1 = np.zeros((n1 * m1, r1))
    V1 = np.zeros((m1 * p1, r1))
    W1 = np.zeros((p1 * n1, r1))
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                c = j1 * p1 + k1
                U1[i1 * m1 + j1, c] = 1.0
                V1[j1 * p1 + k1, c] = 1.0
                W1[k1 * n1 + i1, c] = 1.0
                
    # Decomposition 2: 3x1x2
    n2, m2, p2 = 3, 1, 2
    r2 = m2 * p2
    U2 = np.zeros((n2 * m2, r2))
    V2 = np.zeros((m2 * p2, r2))
    W2 = np.zeros((p2 * n2, r2))
    
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                c = j2 * p2 + k2
                U2[i2 * m2 + j2, c] = 1.0
                V2[j2 * p2 + k2, c] = 1.0
                W2[k2 * n2 + i2, c] = 1.0

    # Kronecker products
    U_kr = np.kron(U1, U2) # Shape (n1*m1*n2*m2, r1*r2) -> (12, 36)
    V_kr = np.kron(V1, V2) # Shape (m1*p1*m2*p2, r1*r2) -> (36, 36)
    W_kr = np.kron(W1, W2) # Shape (p1*n1*p2*n2, r1*r2) -> (18, 36)
    
    # Permutation for U
    # Current row index: u1 * (n2*m2) + u2
    # u1 = i1*m1 + j1
    # u2 = i2*m2 + j2
    # Target row index: i*m + j
    # i = i1*n2 + i2
    # j = j1*m2 + j2
    perm_U = []
    for i1 in range(n1):
        for i2 in range(n2):
            for j1 in range(m1):
                for j2 in range(m2):
                    # Calculate current index in U_kr
                    u1 = i1 * m1 + j1
                    u2 = i2 * m2 + j2
                    curr_idx = u1 * (n2 * m2) + u2
                    # Calculate target index
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    target_idx = i * m + j
                    perm_U.append((curr_idx, target_idx))
                    
    # Apply permutation to U_kr
    # We want U[target_idx] = U_kr[curr_idx]
    # Create an array of indices to sort by target_idx
    # Or just build the new matrix
    U_new = np.zeros_like(U_kr)
    for curr_idx, target_idx in perm_U:
        U_new[target_idx] = U_kr[curr_idx]
        
    # Permutation for V
    # Current row index: v1 * (m2*p2) + v2
    # v1 = j1*p1 + k1
    # v2 = j2*p2 + k2
    # Target row index: j*p + k
    # j = j1*m2 + j2
    # k = k1*p2 + k2
    perm_V = []
    for j1 in range(m1):
        for j2 in range(m2):
            for k1 in range(p1):
                for k2 in range(p2):
                    v1 = j1 * p1 + k1
                    v2 = j2 * p2 + k2
                    curr_idx = v1 * (m2 * p2) + v2
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    target_idx = j * p + k
                    perm_V.append((curr_idx, target_idx))
                    
    V_new = np.zeros_like(V_kr)
    for curr_idx, target_idx in perm_V:
        V_new[target_idx] = V_kr[curr_idx]
        
    # Permutation for W
    # Current row index: w1 * (p2*n2) + w2
    # w1 = k1*n1 + i1
    # w2 = k2*n2 + i2
    # Target row index: k*n + i
    # k = k1*p2 + k2
    # i = i1*n2 + i2
    perm_W = []
    for k1 in range(p1):
        for k2 in range(p2):
            for i1 in range(n1):
                for i2 in range(n2):
                    w1 = k1 * n1 + i1
                    w2 = k2 * n2 + i2
                    curr_idx = w1 * (p2 * n2) + w2
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    target_idx = k * n + i
                    perm_W.append((curr_idx, target_idx))
                    
    W_new = np.zeros_like(W_kr)
    for curr_idx, target_idx in perm_W:
        W_new[target_idx] = W_kr[curr_idx]
        
    # Verify
    # M[idx1, idx2, idx3] = sum_c U[idx1,c] V[idx2,c] W[idx3,c]
    # idx1 = i*m+j, idx2 = j*p+k, idx3 = k*n+i
    
    # Check if time allows for verification
    if time.time() < deadline - 5:
        M_calc = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
        M_true = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_true[i, j, k] = 1.0
        # The tensor M is defined such that M[i,j,k] = 1
        # The decomposition should satisfy:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if valid else 0
        # In our construction, every (i,j,k) is valid.
        # Let's check the max residual
        residual = np.max(np.abs(M_calc - M_true))
        if residual > 1e-9:
             # Fallback to schoolbook if Kronecker fails
             rank = n * m * p
             U = [[0.0] * rank for _ in range(n * m)]
             V = [[0.0] * rank for _ in range(m * p)]
             W = [[0.0] * rank for _ in range(p * n)]
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j][col] = 1.0
                         V[j*p+k][col] = 1.0
                         W[k*n+i][col] = 1.0
                         col += 1
             return {"U": U, "V": V, "W": W}
             
    U_list = [[float(x) for x in row] for row in U_new]
    V_list = [[float(x) for x in row] for row in V_new]
    W_list = [[float(x) for x in row] for row in W_new]
    
    return {"U": U_list, "V": V_list, "W": W_list}
