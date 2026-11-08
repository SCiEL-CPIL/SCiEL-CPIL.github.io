import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strassen coefficients for 2x2x2
    # u: A indices (00, 01, 10, 11)
    # v: B indices (00, 01, 10, 11)
    # w: C indices (00, 01, 10, 11)
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0]
    ]
    
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # 2 parts (A_L, A_R)
    # 4 blocks per part (B cols)
    # 7 Strassen products per block
    
    for c in range(rank):
        part = c // 28
        rem = c % 28
        block = rem // 7
        idx = rem % 7
        
        u_vec = U_strassen[idx]
        v_vec = V_strassen[idx]
        w_vec = W_strassen[idx]
        
        # Fill U
        # Local A indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global i = local_i
        # Global j = part*2 + local_j
        for local_idx, val in enumerate(u_vec):
            if val == 0: continue
            li = local_idx // 2
            lj = local_idx % 2
            gi = li
            gj = part * 2 + lj
            U[gi * m + gj][c] = val
            
        # Fill V
        # Local B indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global j = part*2 + local_j
        # Global k = block*2 + local_k
        for local_idx, val in enumerate(v_vec):
            if val == 0: continue
            lj = local_idx // 2
            lk = local_idx % 2
            gj = part * 2 + lj
            gk = block * 2 + lk
            V[gj * p + gk][c] = val
            
        # Fill W
        # Local C indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global i = local_i
        # Global k = block*2 + local_k
        for local_idx, val in enumerate(w_vec):
            if val == 0: continue
            li = local_idx // 2
            lk = local_idx % 2
            gi = li
            gk = block * 2 + lk
            W[gk * n + gi][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
