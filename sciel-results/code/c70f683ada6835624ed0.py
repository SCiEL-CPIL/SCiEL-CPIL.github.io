import time

def construct():
    n, m, p = 3, 4, 8
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term for a 2x2x2 block
    # local_i, local_j, local_k are 0 or 1
    # U_val, V_val, W_val are the coefficients
    # base_i, base_j, base_k are the global offsets
    def add_term_2x2(base_i, base_j, base_k, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # u_coeffs: dict or list for local (i,j)
        # v_coeffs: dict or list for local (j,k)
        # w_coeffs: dict or list for local (k,i)
        # Actually, for a specific rank-1 term in Strassen,
        # U is a vector in R^4 (2x2), V in R^4, W in R^4.
        # We just set the values.
        
        # Flatten indices
        # U index: (base_i + i) * m + (base_j + j)
        # V index: (base_j + j) * p + (base_k + k)
        # W index: (base_k + k) * n + (base_i + i)
        
        for li in range(2):
            for lj in range(2):
                if u_coeffs[li][lj] != 0:
                    gi = base_i + li
                    gj = base_j + lj
                    U[gi * m + gj][col] = u_coeffs[li][lj]
        
        for lj in range(2):
            for lk in range(2):
                if v_coeffs[lj][lk] != 0:
                    gj = base_j + lj
                    gk = base_k + lk
                    V[gj * p + gk][col] = v_coeffs[lj][lk]
                    
        for lk in range(2):
            for li in range(2):
                if w_coeffs[lk][li] != 0:
                    gk = base_k + lk
                    gi = base_i + li
                    W[gk * n + gi][col] = w_coeffs[lk][li]
                    
        col += 1

    # Strassen 2x2x2 coefficients
    # p1 = (a+d)(e+h)
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 1], [1, 1]]
    w1 = [[1, -1], [1, 1]] # W[k,i] -> w1[0][0]=1 (k=0,i=0), w1[0][1]=-1 (k=0,i=1), w1[1][0]=1 (k=1,i=0), w1[1][1]=1 (k=1,i=1)
    
    # p2 = (c+d)e
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [1, 0]]
    w2 = [[0, 1], [0, 1]] # c10 += p2, c11 += p2. k=0,i=1 -> 1. k=1,i=1 -> 1.
    
    # p3 = a(f-h)
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 1], [0, -1]]
    w3 = [[0, 1], [1, 0]] # c01 += p3, c10 += p3. k=0,i=1 -> 1. k=1,i=0 -> 1.
    
    # p4 = d(g-e)
    u4 = [[0, 0], [0, 1]]
    v4 = [[-1, 0], [1, 0]]
    w4 = [[-1, 0], [1, 1]] # c00 -= p4, c10 += p4, c11 += p4. k=0,i=0 -> -1. k=1,i=0 -> 1. k=1,i=1 -> 1.
    
    # p5 = (a+b)h
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[1, -1], [0, 1]] # c00 += p5, c01 -= p5, c11 += p5. k=0,i=0 -> 1. k=0,i=1 -> -1. k=1,i=1 -> 1.
    
    # p6 = (c-a)(e+f)
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[1, 0], [-1, 0]] # c00 += p6, c11 -= p6. k=0,i=0 -> 1. k=1,i=1 -> -1.
    
    # p7 = (b-d)(g+h)
    u7 = [[0, 1], [0, -1]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[0, 1], [0, 1]] # c01 += p7, c10 += p7. k=0,i=1 -> 1. k=1,i=1 -> 1.
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # 1. Handle first 2 rows of A (i=0,1)
    # A is 3x4. We use rows 0,1.
    # B is 4x8. We split B into top (rows 0,1) and bot (rows 2,3).
    # For A_top (2x4) * B_top (2x8):
    # Split A_top into A_L (cols 0,1) and A_R (cols 2,3).
    # Split B_top into 4 blocks of 2x2.
    
    # A_L * B_top
    # base_i = 0
    # base_j = 0
    for block_idx in range(4):
        base_k = block_idx * 2
        for u, v, w in strassen_terms:
            add_term_2x2(0, 0, base_k, u, v, w)
            
    # A_R * B_bot
    # base_i = 0
    # base_j = 2
    # B_bot rows are 2,3. So base_j for B is 2.
    # Wait, V index is j*p + k.
    # j is global row of B.
    # For B_bot, j=2,3.
    # So base_j in add_term_2x2 should be 2?
    # Yes, base_j is the offset for the 'j' dimension of A and B.
    # A's j is col. B's j is row. They are the same index.
    # So for A_R (cols 2,3), base_j = 2.
    for block_idx in range(4):
        base_k = block_idx * 2
        for u, v, w in strassen_terms:
            add_term_2x2(0, 2, base_k, u, v, w)
            
    # 2. Handle 3rd row of A (i=2)
    # A_3 is 1x4. Split into A_3L (cols 0,1) and A_3R (cols 2,3).
    # A_3L * B_top
    # A_3R * B_bot
    
    def add_term_1x2(base_i, base_j, base_k):
        nonlocal col
        # u: 1x2. u[0][0]=1, u[0][1]=1
        # v: 2x2. v[j][k]=1
        # w: 2x1. w[k][0]=1
        
        # U
        U[(base_i) * m + (base_j + 0)][col] = 1
        U[(base_i) * m + (base_j + 1)][col] = 1
        
        # V
        for lj in range(2):
            for lk in range(2):
                V[(base_j + lj) * p + (base_k + lk)][col] = 1
                
        # W
        for lk in range(2):
            W[(base_k + lk) * n + base_i][col] = 1
            
        col += 1

    # A_3L * B_top
    # base_i = 2, base_j = 0
    for block_idx in range(4):
        base_k = block_idx * 2
        add_term_1x2(2, 0, base_k)
        
    # A_3R * B_bot
    # base_i = 2, base_j = 2
    for block_idx in range(4):
        base_k = block_idx * 2
        add_term_1x2(2, 2, base_k)
        
    return {"U": U, "V": V, "W": W}
