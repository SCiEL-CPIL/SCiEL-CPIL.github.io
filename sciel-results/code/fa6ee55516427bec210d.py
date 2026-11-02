import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # We will use 3 blocks of 2x2x3
    num_blocks = 3
    p_block = 3 # 9 / 3
    
    # Rank for one block is 11
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 11 terms for 2x2x3
    # u: (4,), v: (6,), w: (6,)
    # Indices:
    # u: i*2+j
    # v: j*3+k (local k)
    # w: k*2+i (local k)
    
    terms = []
    
    # 1. M1
    u1 = [0, 0, 0, 0]
    u1[0] = 1; u1[3] = 1 # a00, a11
    v1 = [0, 0, 0, 0, 0, 0]
    v1[0] = 1; v1[3] = 1 # b00, b11 (j=0,k=0; j=1,k=1)
    w1 = [0, 0, 0, 0, 0, 0]
    w1[0] = 1; w1[3] = 1 # C00, C11
    terms.append((u1, v1, w1))
    
    # 2. M2
    u2 = [0, 0, 0, 0]
    u2[2] = 1; u2[3] = 1 # a10, a11
    v2 = [0, 0, 0, 0, 0, 0]
    v2[0] = 1 # b00
    w2 = [0, 0, 0, 0, 0, 0]
    w2[1] = 1; w2[3] = -1 # C10, C11
    terms.append((u2, v2, w2))
    
    # 3. M3
    u3 = [0, 0, 0, 0]
    u3[0] = 1 # a00
    v3 = [0, 0, 0, 0, 0, 0]
    v3[3] = 1; v3[4] = -1 # b11, b10 (j=1,k=1; j=1,k=0)
    w3 = [0, 0, 0, 0, 0, 0]
    w3[2] = 1; w3[3] = 1 # C01, C11
    terms.append((u3, v3, w3))
    
    # 4. M4
    u4 = [0, 0, 0, 0]
    u4[3] = 1 # a11
    v4 = [0, 0, 0, 0, 0, 0]
    v4[0] = 1; v4[1] = -1 # b00, b01
    w4 = [0, 0, 0, 0, 0, 0]
    w4[0] = 1; w4[1] = 1 # C00, C10
    terms.append((u4, v4, w4))
    
    # 5. M5
    u5 = [0, 0, 0, 0]
    u5[0] = 1; u5[1] = 1 # a00, a01
    v5 = [0, 0, 0, 0, 0, 0]
    v5[4] = 1 # b10
    w5 = [0, 0, 0, 0, 0, 0]
    w5[0] = -1; w5[2] = 1 # C00, C01
    terms.append((u5, v5, w5))
    
    # 6. M6
    u6 = [0, 0, 0, 0]
    u6[2] = 1; u6[0] = -1 # a10, a00
    v6 = [0, 0, 0, 0, 0, 0]
    v6[1] = 1; v6[3] = 1 # b01, b11
    w6 = [0, 0, 0, 0, 0, 0]
    w6[3] = -1 # C11
    terms.append((u6, v6, w6))
    
    # 7. M7
    u7 = [0, 0, 0, 0]
    u7[1] = 1; u7[3] = -1 # a01, a11
    v7 = [0, 0, 0, 0, 0, 0]
    v7[0] = 1; v7[4] = 1 # b00, b10
    w7 = [0, 0, 0, 0, 0, 0]
    w7[0] = 1 # C00
    terms.append((u7, v7, w7))
    
    # 8. M8
    u8 = [0, 0, 0, 0]
    u8[0] = 1 # a00
    v8 = [0, 0, 0, 0, 0, 0]
    v8[2] = 1 # b02
    w8 = [0, 0, 0, 0, 0, 0]
    w8[4] = 1 # C02
    terms.append((u8, v8, w8))
    
    # 9. M9
    u9 = [0, 0, 0, 0]
    u9[1] = 1 # a01
    v9 = [0, 0, 0, 0, 0, 0]
    v9[5] = 1 # b12
    w9 = [0, 0, 0, 0, 0, 0]
    w9[4] = 1 # C02
    terms.append((u9, v9, w9))
    
    # 10. M10
    u10 = [0, 0, 0, 0]
    u10[2] = 1 # a10
    v10 = [0, 0, 0, 0, 0, 0]
    v10[2] = 1 # b02
    w10 = [0, 0, 0, 0, 0, 0]
    w10[5] = 1 # C12
    terms.append((u10, v10, w10))
    
    # 11. M11
    u11 = [0, 0, 0, 0]
    u11[3] = 1 # a11
    v11 = [0, 0, 0, 0, 0, 0]
    v11[5] = 1 # b12
    w11 = [0, 0, 0, 0, 0, 0]
    w11[5] = 1 # C12
    terms.append((u11, v11, w11))
    
    # Fill U, V, W for 3 blocks
    for b in range(num_blocks):
        offset = b * rank_per_block
        for r in range(rank_per_block):
            col = offset + r
            u, v, w = terms[r]
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    U[idx_u][col] = u[idx_u]
            
            # Fill V
            for j in range(m):
                for k_local in range(p_block):
                    idx_v_local = j * p_block + k_local
                    k_global = b * p_block + k_local
                    idx_v = j * p + k_global
                    V[idx_v][col] = v[idx_v_local]
            
            # Fill W
            for k_local in range(p_block):
                for i in range(n):
                    idx_w_local = k_local * n + i
                    k_global = b * p_block + k_local
                    idx_w = k_global * n + i
                    W[idx_w][col] = w[idx_w_local]
                    
    return {"U": U, "V": V, "W": W}
