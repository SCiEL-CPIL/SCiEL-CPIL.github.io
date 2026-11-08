def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    n, m, p = 2, 3, 8
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for k_half in [0, 4]:
        for k0 in [k_half, k_half + 2]:
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li*2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la*2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk*2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        
        U[2][col] = 1
        U[5][col] = 1
        for dk in range(4):
            V[16 + k_half + dk][col] = 1
            W[2*(k_half + dk)][col] = 1
            W[2*(k_half + dk) + 1][col] = 1
        col += 1
    
    # Validate
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    uv = U[i*m+a][c]
                    vv = V[a*p+k][c]
                    ww = W[k*n+i][c]
                    if uv != 0 and vv != 0 and ww != 0:
                        acc += uv * vv * ww
                if acc != 1:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col_sb = 0
                    for i2 in range(n):
                        for a2 in range(m):
                            for k2 in range(p):
                                U_sb[i2*m+a2][col_sb] = 1
                                V_sb[a2*p+k2][col_sb] = 1
                                W_sb[k2*n+i2][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
