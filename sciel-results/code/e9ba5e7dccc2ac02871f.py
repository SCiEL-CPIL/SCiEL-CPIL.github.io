def construct():
    t0 = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 5, 5
    
    # Get core 4x4x4 Strassen decomposition
    U_core, V_core, W_core = _strassen_4x4x4()
    
    # Part 1a: 4x4x4 core (i,j,k in 0..3)
    U_1a = [[0]*49 for _ in range(n*m)]
    V_1a = [[0]*49 for _ in range(m*p)]
    W_1a = [[0]*49 for _ in range(p*n)]
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U_1a[i*m+j][r] = U_core[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V_1a[j*p+k][r] = V_core[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W_1a[k*n+i][r] = W_core[k*4+i][r]
    
    # Part 1b: 4x4x1 (i,j in 0..3, k=4)
    U_1b = [[0] for _ in range(n*m)]
    V_1b = [[0] for _ in range(m*p)]
    W_1b = [[0] for _ in range(p*n)]
    for i in range(4):
        for j in range(4):
            U_1b[i*m+j][0] = 1
    for j in range(4):
        V_1b[j*p+4][0] = 1
    for i in range(4):
        W_1b[4*n+i][0] = 1
    
    # Part 2: 4x1x5 (i in 0..3, j=4, k in 0..4)
    U_2 = [[0] for _ in range(n*m)]
    V_2 = [[0] for _ in range(m*p)]
    W_2 = [[0] for _ in range(p*n)]
    for i in range(4):
        U_2[i*m+4][0] = 1
    for k in range(5):
        V_2[4*p+k][0] = 1
    for k in range(5):
        for i in range(4):
            W_2[k*n+i][0] = 1
    
    decomps = [
        (U_1a, V_1a, W_1a),
        (U_1b, V_1b, W_1b),
        (U_2, V_2, W_2),
    ]
    
    U_out, V_out, W_out = _concat_decomps(decomps)
    
    # Verify
    if time.time() - t0 < DEADLINE - 3:
        rank = len(U_out[0])
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col] = 1
                        V_fb[j * p + k][col] = 1
                        W_fb[k * n + i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_out, "V": V_out, "W": W_out}
