def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col_idx] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col_idx] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col_idx] += w_vec[i]

    # Define the 11 terms for 2x3x2 block
    # u is length 6 (A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2])
    # v is length 6 (B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1])
    # w is length 4 (C[0,0], C[0,1], C[1,0], C[1,1]) -> Note: W index is k*n+i.
    # Local W indices:
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    
    # Part 1: Outer product (4 terms)
    # 1. A[0,0] * B[0,0] -> C[0,0]
    u1 = [1, 0, 0, 0, 0, 0]
    v1 = [1, 0, 0, 0, 0, 0]
    w1 = [1, 0, 0, 0]
    
    # 2. A[0,0] * B[0,1] -> C[0,1]
    u2 = [1, 0, 0, 0, 0, 0]
    v2 = [0, 1, 0, 0, 0, 0]
    w2 = [0, 0, 0, 1] # C[0,1] is index 2? No.
    # W index = k*n + i.
    # C[0,1] means i=0, k=1. Index = 1*2 + 0 = 2.
    # My w1 was C[0,0] (i=0, k=0) -> 0.
    # w2 should be C[0,1] (i=0, k=1) -> 2.
    w2 = [0, 0, 1, 0]
    
    # 3. A[1,0] * B[0,0] -> C[1,0]
    u3 = [0, 0, 0, 1, 0, 0]
    v3 = [1, 0, 0, 0, 0, 0]
    # C[1,0] means i=1, k=0. Index = 0*2 + 1 = 1.
    w3 = [0, 1, 0, 0]
    
    # 4. A[1,0] * B[0,1] -> C[1,1]
    u4 = [0, 0, 0, 1, 0, 0]
    v4 = [0, 1, 0, 0, 0, 0]
    # C[1,1] means i=1, k=1. Index = 1*2 + 1 = 3.
    w4 = [0, 0, 0, 1]

    # Part 2: Strassen (7 terms)
    # M1 = (A[0,1]+A[1,2]) * (B[1,0]+B[2,1])
    # M2 = (A[1,1]+A[1,2]) * B[1,0]
    # M3 = A[0,1] * (B[1,1]-B[2,1])
    # M4 = A[1,2] * (B[2,0]-B[1,0])
    # M5 = (A[0,1]+A[0,2]) * B[2,1]
    # M6 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    # M7 = (A[0,2]-A[1,2]) * (B[2,0]+B[2,1])
    
    # Let's map Strassen variables to local indices
    # a11=A[0,1] (idx 1), a12=A[0,2] (idx 2)
    # a21=A[1,1] (idx 4), a22=A[1,2] (idx 5)
    # b11=B[1,0] (idx 2), b12=B[1,1] (idx 3)
    # b21=B[2,0] (idx 4), b22=B[2,1] (idx 5)
    
    # M1
    u5 = [0, 1, 0, 0, 0, 1] # A[0,1]+A[1,2]
    v5 = [0, 0, 1, 0, 0, 1] # B[1,0]+B[2,1]
    # Contributes to:
    # C[0,0] (idx 0): +1
    # C[1,1] (idx 3): +1
    w5 = [1, 0, 0, 1]
    
    # M2
    u6 = [0, 0, 0, 0, 1, 1] # A[1,1]+A[1,2]
    v6 = [0, 0, 1, 0, 0, 0] # B[1,0]
    # Contributes to:
    # C[1,0] (idx 1): +1
    w6 = [0, 1, 0, 0]
    
    # M3
    u7 = [0, 1, 0, 0, 0, 0] # A[0,1]
    v7 = [0, 0, 0, 1, 0, -1] # B[1,1]-B[2,1]
    # Contributes to:
    # C[0,1] (idx 2): +1
    # C[1,1] (idx 3): -1
    w7 = [0, 0, 1, -1]
    
    # M4
    u8 = [0, 0, 0, 0, 0, 1] # A[1,2]
    v8 = [0, 0, -1, 0, 1, 0] # B[2,0]-B[1,0]
    # Contributes to:
    # C[0,0] (idx 0): -1
    # C[0,1] (idx 2): +1
    w8 = [-1, 0, 1, 0]
    
    # M5
    u9 = [0, 1, 1, 0, 0, 0] # A[0,1]+A[0,2]
    v9 = [0, 0, 0, 0, 0, 1] # B[2,1]
    # Contributes to:
    # C[0,0] (idx 0): +1
    w9 = [1, 0, 0, 0]
    
    # M6
    u10 = [0, -1, 0, 0, 1, 0] # A[1,1]-A[0,1]
    v10 = [0, 0, 1, 1, 0, 0] # B[1,0]+B[1,1]
    # Contributes to:
    # C[1,0] (idx 1): +1
    # C[1,1] (idx 3): +1
    w10 = [0, 1, 0, 1]
    
    # M7
    u11 = [0, 0, 1, 0, 0, -1] # A[0,2]-A[1,2]
    v11 = [0, 0, 0, 0, 1, 1] # B[2,0]+B[2,1]
    # Contributes to:
    # C[0,0] (idx 0): +1
    # C[0,1] (idx 2): +1
    w11 = [1, 0, 1, 0]

    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
        (u8, v8, w8),
        (u9, v9, w9),
        (u10, v10, w10),
        (u11, v11, w11)
    ]

    # Assemble 5 blocks
    for block in range(5):
        for t_idx in range(11):
            col = block * 11 + t_idx
            u, v, w = terms[t_idx]
            
            # Map U (same for all blocks)
            for i in range(6):
                if u[i] != 0:
                    U[i][col] = u[i]
                    
            # Map V
            # Local v index j*2 + k_local
            # Global V index j*10 + k_global
            # k_global = block*2 + k_local
            for j in range(3):
                for k_local in range(2):
                    val = v[j*2 + k_local]
                    if val != 0:
                        k_global = block * 2 + k_local
                        V[j*10 + k_global][col] = val
                        
            # Map W
            # Local w index k_local*2 + i
            # Global W index k_global*2 + i
            # k_global = block*2 + k_local
            for k_local in range(2):
                for i in range(2):
                    val = w[k_local*2 + i]
                    if val != 0:
                        k_global = block * 2 + k_local
                        W[k_global*2 + i][col] = val
                        
    return {"U": U, "V": V, "W": W}
