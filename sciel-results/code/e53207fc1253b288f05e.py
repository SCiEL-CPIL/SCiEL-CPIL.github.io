import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank calculation:
    # 8 blocks of 2x2x2 (rank 7 each) -> 56
    # 1 slice j=4 (rank 1) -> 1
    # 1 slice k=4, j in 0..3 (rank 4) -> 4
    # Total 61
    rank = 61
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Strassen vectors for 2x2x2
    # u, v, w are tuples of 2 elements
    strassen_terms = [
        ((1, 1), (1, 1), (1, 1)),
        ((0, 1), (1, 1), (1, 0)),
        ((1, 0), (1, -1), (0, 1)),
        ((0, 1), (-1, 1), (1, 0)),
        ((1, 0), (1, 1), (0, 1)),
        ((-1, 1), (1, 0), (1, 1)),
        ((1, -1), (0, 1), (1, 1))
    ]
    
    # Define 8 blocks
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((
                    [i_base, i_base + 1],
                    [j_base, j_base + 1],
                    [k_base, k_base + 1]
                ))
    
    # Fill Strassen terms
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        for u, v, w in strassen_terms:
            U[i1 * m + j1][col] += u[0]
            U[i2 * m + j1][col] += u[1]
            U[i1 * m + j2][col] += u[0]
            U[i2 * m + j2][col] += u[1]
            
            V[j1 * p + k1][col] += v[0]
            V[j2 * p + k1][col] += v[1]
            V[j1 * p + k2][col] += v[0]
            V[j2 * p + k2][col] += v[1]
            
            W[k1 * n + i1][col] += w[0]
            W[k1 * n + i2][col] += w[1]
            W[k2 * n + i1][col] += w[0]
            W[k2 * n + i2][col] += w[1]
            
            col += 1
    
    # Slice j=4
    # U_i = A_i4, V_4 = 1, W_k = B_4k
    # We use symbolic 1s and -1s? No, we need exact decomposition.
    # The tensor T_ijk = delta_ik.
    # For j=4, T_i4k = delta_ik.
    # We need U_i W_k = delta_ik.
    # This is rank 4 (identity matrix).
    # So we need 4 terms, not 1.
    
    # Correction:
    # T_ijk = delta_ik.
    # For j=4, T_i4k = delta_ik.
    # Matrix I_4. Rank 4.
    # So we need 4 terms for j=4 slice.
    
    # For k=4, j in 0..3.
    # T_ij4 = delta_ij.
    # Matrix I_4. Rank 4.
    # So we need 4 terms for k=4 slice.
    
    # Total rank = 56 + 4 + 4 = 64.
    
    # Let's adjust rank
    rank = 64
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    
    # Refill Strassen
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        for u, v, w in strassen_terms:
            U[i1 * m + j1][col] += u[0]
            U[i2 * m + j1][col] += u[1]
            U[i1 * m + j2][col] += u[0]
            U[i2 * m + j2][col] += u[1]
            
            V[j1 * p + k1][col] += v[0]
            V[j2 * p + k1][col] += v[1]
            V[j1 * p + k2][col] += v[0]
            V[j2 * p + k2][col] += v[1]
            
            W[k1 * n + i1][col] += w[0]
            W[k1 * n + i2][col] += w[1]
            W[k2 * n + i1][col] += w[0]
            W[k2 * n + i2][col] += w[1]
            
            col += 1
    
    # Slice j=4 (T_i4k = delta_ik)
    # 4 terms for identity
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + i][col] = 1  # Wait, V index is j*p+k. j=4, k=i?
        # T_i4k = delta_ik.
        # Term: U_i * V_4k * W_ki?
        # W index is k*n+i.
        # We want U_i V_4k W_ki = delta_ik.
        # Let U_i = 1, V_4k = delta_ki, W_ki = 1.
        # V index: j=4, k=i. So V[4*p + i].
        # W index: k=i, i=i. So W[i*n + i].
        
        U[i * m + 4][col] = 1
        V[4 * p + i][col] = 1
        W[i * n + i][col] = 1
        col += 1
        
    # Slice k=4, j in 0..3 (T_ij4 = delta_ij)
    # 4 terms for identity
    for j in range(4):
        # T_ij4 = delta_ij.
        # Term: U_ij * V_j4 * W_4i?
        # W index: k=4, i=j. So W[4*n + j].
        # V index: j=j, k=4. So V[j*p + 4].
        # U index: i=j, j=j. So U[j*m + j].
        
        U[j * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + j][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
