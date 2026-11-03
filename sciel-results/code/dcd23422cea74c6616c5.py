def construct():
    n, m, p = 2, 4, 6
    import numpy as np
    
    # Define 2x2x2x2 factors and verify
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [0,1,-1,0], [0,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        ([0,-1,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [0,0,1,-1]),
    ]
    
    # Verify 2x2x2x2
    T = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1
    
    T_recon = np.zeros((4,4,4))
    for U_v, V_v, W_v in strassen_2x2:
        U_a = np.array(U_v, dtype=float)
        V_a = np.array(V_v, dtype=float)
        W_a = np.array(W_v, dtype=float)
        T_recon += np.einsum('i,j,k->ijk', U_a, V_a, W_a)
    
    use_strassen = np.allclose(T_recon, T, atol=1e-10)
    
    if use_strassen:
        factors = strassen_2x2
    else:
        # Schoolbook 2x2x2x2
        factors = [
            ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
            ([0,0,1,0], [0,0,1,0], [1,0,0,0]),
            ([1,0,0,0], [0,1,0,0], [0,0,1,0]),
            ([0,0,1,0], [0,0,0,1], [0,0,1,0]),
            ([0,0,0,1], [1,0,0,0], [0,1,0,0]),
            ([0,1,0,0], [0,0,1,0], [0,1,0,0]),
            ([0,0,0,1], [0,1,0,0], [0,0,0,1]),
            ([0,1,0,0], [0,0,0,1], [0,0,0,1]),
        ]
    
    num_blocks = 2 * 3
    rank = len(factors) * num_blocks
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for mb in range(2):
        for pb in range(3):
            for U_v, V_v, W_v in factors:
                for idx in range(4):
                    i_l = idx // 2
                    j_l = idx % 2
                    u_row = i_l * m + (mb * 2 + j_l)
                    U[u_row][col] = U_v[idx]
                
                for idx in range(4):
                    j_l = idx // 2
                    k_l = idx % 2
                    v_row = (mb * 2 + j_l) * p + (pb * 2 + k_l)
                    V[v_row][col] = V_v[idx]
                
                for idx in range(4):
                    k_l = idx // 2
                    i_l = idx % 2
                    w_row = (pb * 2 + k_l) * n + i_l
                    W[w_row][col] = W_v[idx]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
