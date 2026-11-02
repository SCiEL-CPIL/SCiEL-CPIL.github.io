# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize U, V, W as lists of lists
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U: i*m + j (0:00, 1:01, 2:10, 3:11)
    # V: j*p + k (0:00, 1:01, 2:10, 3:11) -- Wait, V is for B. B is 2x2 in block.
    # Local V indices: 0:00, 1:01, 2:10, 3:11
    # W: k*n + i (0:00, 1:10, 2:01, 3:11) -- Wait, W is for C.
    # Local W indices: 0:00, 1:10, 2:01, 3:11?
    # Prompt says W[k*n+i].
    # For 2x2 C:
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    # So local W indices: 0:C00, 1:C10, 2:C01, 3:C11.
    
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V local indices: 0:00, 1:01, 2:10, 3:11
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W local indices: 0:C00, 1:C10, 2:C01, 3:C11
    sW = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for b in range(3):
        for r in range(7):
            # U
            for i in range(4):
                U[i][col] += sU[r][i]
            
            # V
            # Map local V index to global
            # Local 0 -> j=0, k=2b
            # Local 1 -> j=0, k=2b+1
            # Local 2 -> j=1, k=2b
            # Local 3 -> j=1, k=2b+1
            for lv in range(4):
                coeff = sV[r][lv]
                if coeff != 0:
                    j, k_loc = divmod(lv, 2) # 0:0,0 1:0,1 2:1,0 3:1,1
                    k_glob = 2 * b + k_loc
                    v_idx = j * p + k_glob
                    V[v_idx][col] += coeff
            
            # W
            # Map local W index to global
            # Local 0 -> C00 -> k=2b, i=0
            # Local 1 -> C10 -> k=2b, i=1
            # Local 2 -> C01 -> k=2b+1, i=0
            # Local 3 -> C11 -> k=2b+1, i=1
            for lw in range(4):
                coeff = sW[r][lw]
                if coeff != 0:
                    # lw: 0,1,2,3
                    # k_loc: 0,0,1,1
                    # i_loc: 0,1,0,1
                    k_loc = lw // 2
                    i_loc = lw % 2
                    k_glob = 2 * b + k_loc
                    w_idx = k_glob * n + i_loc
                    W[w_idx][col] += coeff
            
            col += 1
            
    # Schoolbook for last block (k=6)
    # 4 terms
    sb_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    sb_V = [
        [1, 0], # B00
        [0, 1], # B01? No, B is 2x1.
        # B indices: 0:00, 1:10
        [1, 0],
        [0, 1]
    ]
    # Wait, B is 2x7. Last column is index 6.
    # B[0,6] is local 0. B[1,6] is local 1.
    sb_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    sb_W = [
        [1, 0], # C00
        [0, 1], # C10? No, C is 2x1.
        # C indices: 0:00, 1:10
        [1, 0],
        [0, 1]
    ]
    # W indices: 0:C00, 1:C10
    sb_W = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += sb_U[r][i]
        for lv in range(2):
            coeff = sb_V[r][lv]
            if coeff != 0:
                j = lv
                k_glob = 6
                v_idx = j * p + k_glob
                V[v_idx][col] += coeff
        for lw in range(2):
            coeff = sb_W[r][lw]
            if coeff != 0:
                i_loc = lw
                k_glob = 6
                w_idx = k_glob * n + i_loc
                W[w_idx][col] += coeff
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
