def construct():
    n, m, p = 2, 4, 6
    # Split j into two groups of 2, k into three groups of 2
    # 6 blocks, each Strassen 2x2x2x2, rank 7 each
    # Total rank = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for g in range(2):  # A group: 0 or 1
        for b in range(3):  # B column block: 0, 1, 2
            # Strassen terms for 2x2 x 2x2
            # A_g = A[:, 2g:2g+2], B_g_b = B[2g:2g+2, 2b:2b+2]
            # A_g[0,0]=A[0,2g], A_g[0,1]=A[0,2g+1]
            # A_g[1,0]=A[1,2g], A_g[1,1]=A[1,2g+1]
            # B_g_b[0,0]=B[2g,2b], B_g_b[0,1]=B[2g,2b+1]
            # B_g_b[1,0]=B[2g+1,2b], B_g_b[1,1]=B[2g+1,2b+1]
            
            # P1 = (A_g[0,0]+A_g[1,1]) * (B_g_b[0,0]+B_g_b[1,0])
            # U: A[0,2g], A[1,2g+1]
            # V: B[2g,2b], B[2g+1,2b]
            # W: C[0,2b] (+1), C[1,2b+1] (+1)
            U[0 * m + 2*g][col] = 1
            U[1 * m + 2*g+1][col] = 1
            V[2*g * p + 2*b][col] = 1
            V[(2*g+1) * p + 2*b][col] = 1
            W[2*b * n + 0][col] = 1   # C[0, 2b]
            W[(2*b+1) * n + 1][col] = 1  # C[1, 2b+1]
            col += 1
            
            # P2 = (A_g[1,0]+A_g[1,1]) * (B_g_b[0,0]+B_g_b[1,0])
            # U: A[1,2g], A[1,2g+1]
            # V: B[2g,2b], B[2g+1,2b]
            # W: C[1,2b] (+1)
            U[1 * m + 2*g][col] = 1
            U[1 * m + 2*g+1][col] = 1
            V[2*g * p + 2*b][col] = 1
            V[(2*g+1) * p + 2*b][col] = 1
            W[2*b * n + 1][col] = 1   # C[1, 2b]
            col += 1
            
            # P3 = A_g[0,0] * (B_g_b[0,1]-B_g_b[1,0])
            # U: A[0,2g]
            # V: B[2g,2b+1] (+1), B[2g+1,2b] (-1)
            # W: C[0,2b+1] (+1), C[1,2b+1] (+1)
            U[0 * m + 2*g][col] = 1
            V[2*g * p + 2*b+1][col] = 1
            V[(2*g+1) * p + 2*b][col] = -1
            W[(2*b+1) * n + 0][col] = 1   # C[0, 2b+1]
            W[(2*b+1) * n + 1][col] = 1   # C[1, 2b+1]
            col += 1
            
            # P4 = A_g[1,1] * (B_g_b[0,1]-B_g_b[1,0])
            # U: A[1,2g+1]
            # V: B[2g,2b+1] (+1), B[2g+1,2b] (-1)
            # W: C[0,2b] (+1)
            U[1 * m + 2*g+1][col] = 1
            V[2*g * p + 2*b+1][col] = 1
            V[(2*g+1) * p + 2*b][col] = -1
            W[2*b * n + 0][col] = 1   # C[0, 2b]
            col += 1
            
            # P5 = (A_g[0,0]+A_g[0,1]) * (B_g_b[1,1]-B_g_b[1,0])
            # U: A[0,2g], A[0,2g+1]
            # V: B[2g+1,2b+1] (+1), B[2g+1,2b] (-1)
            # W: C[0,2b] (-1), C[0,2b+1] (+1)
            U[0 * m + 2*g][col] = 1
            U[0 * m + 2*g+1][col] = 1
            V[(2*g+1) * p + 2*b+1][col] = 1
            V[(2*g+1) * p + 2*b][col] = -1
            W[2*b * n + 0][col] = -1   # C[0, 2b]
            W[(2*b+1) * n + 0][col] = 1   # C[0, 2b+1]
            col += 1
            
            # P6 = (A_g[1,0]-A_g[0,1]) * (B_g_b[0,0]+B_g_b[0,1])
            # U: A[1,2g] (+1), A[0,2g+1] (-1)
            # V: B[2g,2b] (+1), B[2g,2b+1] (+1)
            # W: C[1,2b+1] (+1)
            U[1 * m + 2*g][col] = 1
            U[0 * m + 2*g+1][col] = -1
            V[2*g * p + 2*b][col] = 1
            V[2*g * p + 2*b+1][col] = 1
            W[(2*b+1) * n + 1][col] = 1   # C[1, 2b+1]
            col += 1
            
            # P7 = (A_g[1,1]-A_g[1,0]) * (B_g_b[1,1]-B_g_b[0,1])
            # U: A[1,2g+1] (+1), A[1,2g] (-1)
            # V: B[2g+1,2b+1] (+1), B[2g,2b+1] (-1)
            # W: C[0,2b] (-1), C[1,2b] (+1)
            U[1 * m + 2*g+1][col] = 1
            U[1 * m + 2*g][col] = -1
            V[(2*g+1) * p + 2*b+1][col] = 1
            V[2*g * p + 2*b+1][col] = -1
            W[2*b * n + 0][col] = -1   # C[0, 2b]
            W[2*b * n + 1][col] = 1   # C[1, 2b]
            col += 1
    
    return {"U": U, "V": V, "W": W}
