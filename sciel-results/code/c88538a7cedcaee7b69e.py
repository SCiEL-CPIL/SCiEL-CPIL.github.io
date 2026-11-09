DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # u: (i, j), v: (j, k), w: (k, i)
    strassen_terms = [
        # (U, V, W)
        # r=0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # r=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):1, (1,0):1}),
        # r=2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (1,0):1}),
        # r=3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        # r=4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # r=5
        ({(1,1):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
        # r=6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    ]
    
    # Map to 2x2x4 (Rank 7)
    # Then two of these for 2x4x4 (Rank 14)
    
    for part in range(2):
        base_r = part * 7
        j_offset = part * 2
        
        for local_r in range(7):
            r = base_r + local_r
            u_dict, v_dict, w_dict = strassen_terms[local_r]
            
            for (i, j_local), val in u_dict.items():
                j_global = j_offset + j_local
                U[i * m + j_global][r] += val
                
            for (j_local, k_local), val in v_dict.items():
                # k_local is 0 or 1 in Strassen 2x2x2
                # In 2x2x4, k is 0..3.
                # We map k_local to k_global = k_local?
                # Wait, Strassen 2x2x2 covers k=0,1.
                # For 2x2x4, we need k=0..3.
                # The "sharing" trick means we use the SAME v_dict for k=0,1 and k=2,3?
                # No, B is 2x4. B_L is 2x2, B_R is 2x2.
                # Strassen for B_L uses B_{00}, B_{01}, B_{10}, B_{11}.
                # Strassen for B_R uses B'_{00}, ...
                # The coefficients V_{j,k} are the same.
                # So V_{j, 0}^{(r)} = V_{j, 0}^{Strassen}
                # V_{j, 1}^{(r)} = V_{j, 1}^{Strassen}
                # V_{j, 2}^{(r)} = V_{j, 0}^{Strassen}
                # V_{j, 3}^{(r)} = V_{j, 1}^{Strassen}
                
                # So we add to k_local and k_local+2
                V[j_global * p + k_local][r] += val
                V[j_global * p + k_local + 2][r] += val
                
            for (k_local, i_local), val in w_dict.items():
                # W_{k,i}
                # k_local is 0,1.
                # We need W for k=0,1 and k=2,3.
                # The output C is 2x4.
                # C_L is 2x2, C_R is 2x2.
                # Strassen W for C_L maps to i.
                # Strassen W for C_R maps to i.
                # So W_{k_local, i} and W_{k_local+2, i} get the same val.
                
                W[k_local * n + i_local][r] += val
                W[(k_local + 2) * n + i_local][r] += val

    return {"U": U, "V": V, "W": W}
