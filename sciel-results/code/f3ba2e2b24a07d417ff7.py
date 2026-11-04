import time

def construct():
    t_start = time.time()
    limit = t_start + 140.0
    
    n, m, p = 2, 3, 8
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: (a00, a01, a10, a11)
    su = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]
    ]
    # V: (b00, b01, b10, b11)
    sv = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]
    ]
    # W: (c00, c01, c10, c11)
    sw = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]
    ]
    
    col = 0
    # 4 blocks for j=0,1
    for b_idx in range(4):
        k_offset = b_idx * 2
        for s_idx in range(7):
            # U: rows i*3+j for i in 0,1, j in 0,1
            for i in range(2):
                for j in range(2):
                    val = su[s_idx][i*2 + j]
                    if val:
                        U[i*m + j][col] += val
                        
            # V: rows j*8+k for j in 0,1, k in k_offset, k_offset+1
            for j in range(2):
                for k in range(2):
                    val = sv[s_idx][j*2 + k]
                    if val:
                        V[j*p + (k_offset + k)][col] += val
                        
            # W: rows k*2+i for k in k_offset, k_offset+1, i in 0,1
            for k in range(2):
                for i in range(2):
                    val = sw[s_idx][k*2 + i]
                    if val:
                        W[(k_offset + k)*n + i][col] += val
            col += 1
            
            if time.time() > limit:
                # Fallback
                rb = n*m*p
                U_f = [[0]*rb for _ in range(n*m)]
                V_f = [[0]*rb for _ in range(m*p)]
                W_f = [[0]*rb for _ in range(p*n)]
                c = 0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            U_f[i*m+a][c] = 1
                            V_f[a*p+k][c] = 1
                            W_f[k*n+i][c] = 1
                            c += 1
                return {"U": U_f, "V": V_f, "W": W_f}

    # Part 2: j=2, rank 1
    # U[i, 2, col] = 1
    # V[2, k, col] = 1
    # W[k, i, col] = 1
    for i in range(n):
        U[i*m + 2][col] = 1
    for k in range(p):
        V[2*p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][col] = 1
    col += 1

    # Validation
    valid = all(
        sum(U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] for r in range(rank)) == 1
        for i in range(n)
        for a in range(m)
        for k in range(p)
    )
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rb = n*m*p
        U_f = [[0]*rb for _ in range(n*m)]
        V_f = [[0]*rb for _ in range(m*p)]
        W_f = [[0]*rb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i*m+a][c] = 1
                    V_f[a*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
