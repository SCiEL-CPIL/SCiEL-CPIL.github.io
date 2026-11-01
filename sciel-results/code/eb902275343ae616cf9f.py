import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = 2, 4, 6
    
    # Strassen local matrices
    # U_local: 4x7 (rows: 00, 01, 10, 11)
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V_local: 4x7 (rows: 00, 01, 10, 11)
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W_local: 4x7 (rows: 00, 01, 10, 11 corresponding to k,i pairs: (0,0), (0,1), (1,0), (1,1))
    # Note: W index is k*n + i. Here n=2 locally? No, W index is global k*n + i.
    # In local context, we just map local (k, i) to the 4 slots.
    # Slot 0: k=0, i=0
    # Slot 1: k=0, i=1
    # Slot 2: k=1, i=0
    # Slot 3: k=1, i=1
    W_local = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, 0, 0, 0, 0, 1, 0]   # C11
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    # Blocks: i_off in [0], j_off in [0, 2], k_off in [0, 2, 4]
    for i_off in range(0, n, 2):
        for j_off in range(0, m, 2):
            for k_off in range(0, p, 2):
                # Iterate through 7 Strassen terms
                for r in range(7):
                    current_col = col_offset + r
                    
                    # Fill U
                    for i_loc in range(2):
                        for j_loc in range(2):
                            u_val = U_local[i_loc * 2 + j_loc][r]
                            if u_val != 0:
                                u_idx = (i_off + i_loc) * m + (j_off + j_loc)
                                U[u_idx][current_col] = u_val
                                
                    # Fill V
                    for j_loc in range(2):
                        for k_loc in range(2):
                            v_val = V_local[j_loc * 2 + k_loc][r]
                            if v_val != 0:
                                v_idx = (j_off + j_loc) * p + (k_off + k_loc)
                                V[v_idx][current_col] = v_val
                                
                    # Fill W
                    for k_loc in range(2):
                        for i_loc in range(2):
                            w_val = W_local[k_loc * 2 + i_loc][r]
                            if w_val != 0:
                                w_idx = (k_off + k_loc) * n + (i_off + i_loc)
                                W[w_idx][current_col] = w_val
                                
                col_offset += 7
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
