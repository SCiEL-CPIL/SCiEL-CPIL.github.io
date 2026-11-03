import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 vectors
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    v_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    w_strassen = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 1, 0, -1], # M2 -> C10, C11
        [0, 0, 1, 1], # M3 -> C01, C11
        [1, 1, 0, 0], # M4 -> C00, C10
        [-1, 0, 0, 0], # M5 -> C00
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            us = u_strassen[s]
            vt = v_strassen[t] # Wait, v is for B, so v_s for block, v_t for inner
            # Correction:
            # U depends on u_s (block) and u_t (inner)
            # V depends on v_s (block) and v_t (inner)
            # W depends on w_s (block) and w_t (inner)
            
            vs = v_strassen[s]
            ws = w_strassen[s]
            wt = w_strassen[t]
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    ih = i // 2
                    il = i % 2
                    jh = j // 2
                    jl = j % 2
                    idx_u_block = ih * 2 + jh
                    idx_u_inner = il * 2 + jl
                    U[i * 4 + j][col] = us[idx_u_block] * u_strassen[t][idx_u_inner]
                    
            # Fill V
            for j in range(4):
                for k in range(4):
                    jh = j // 2
                    jl = j % 2
                    kh = k // 2
                    kl = k % 2
                    idx_v_block = jh * 2 + kh
                    idx_v_inner = jl * 2 + kl
                    V[j * 4 + k][col] = vs[idx_v_block] * v_strassen[t][idx_v_inner]
                    
            # Fill W
            for k in range(4):
                for i in range(4):
                    kh = k // 2
                    kl = k % 2
                    ih = i // 2
                    il = i % 2
                    idx_w_block = kh * 2 + ih
                    idx_w_inner = kl * 2 + il
                    W[k * 4 + i][col] = ws[idx_w_block] * wt[idx_w_inner]
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
