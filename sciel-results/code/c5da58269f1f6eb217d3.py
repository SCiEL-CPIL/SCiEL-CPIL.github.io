import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A factors: (i, j_local, coeff)
    # 0: (0,0,1), (1,1,1)
    # 1: (1,0,1), (1,1,1)
    # 2: (0,0,1)
    # 3: (1,1,1)
    # 4: (0,0,1), (0,1,1)
    # 5: (1,0,1), (0,0,-1)
    # 6: (0,1,1), (1,1,-1)
    
    A_factors = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    # B factors: (j_local, k_local, coeff)
    # 0: (0,0,1), (1,1,1)
    # 1: (0,0,1)
    # 2: (0,1,1), (1,1,-1)
    # 3: (1,0,1), (0,0,-1)
    # 4: (1,1,1)
    # 5: (0,0,1), (0,1,1)
    # 6: (1,0,1), (1,1,1)
    
    B_factors = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    # C contributions: (i, k_local, coeff)
    # C_00 = M_0 + M_3 - M_4 + M_6
    # C_01 = M_2 + M_4
    # C_10 = M_1 + M_3
    # C_11 = M_0 - M_1 + M_2 + M_5
    
    # We need to add to W for each term s.
    # W contribution for term s is determined by which C_ij it affects.
    # It's easier to iterate s and add to W for the affected C_ij.
    
    # Let's precompute W contributions for each s
    # W_factors[s] is list of (i, k_local, coeff)
    W_factors = [
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)], # M_0 -> C_00, C_11
        [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, -1)], # M_1 -> C_10, C_11
        [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)],  # M_2 -> C_01, C_11
        [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)],  # M_3 -> C_00, C_10
        [(0, 0, -1), (0, 1, 1), (1, 0, 0), (1, 1, 0)], # M_4 -> C_00, C_01
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],  # M_5 -> C_11
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)]   # M_6 -> C_00
    ]
    
    # Wait, let's re-verify W_factors.
    # C_00: M_0, M_3, -M_4, M_6
    # C_01: M_2, M_4
    # C_10: M_1, M_3
    # C_11: M_0, -M_1, M_2, M_5
    
    # s=0 (M_0): C_00 (+1), C_11 (+1)
    # s=1 (M_1): C_10 (+1), C_11 (-1)
    # s=2 (M_2): C_01 (+1), C_11 (+1)
    # s=3 (M_3): C_00 (+1), C_10 (+1)
    # s=4 (M_4): C_00 (-1), C_01 (+1)
    # s=5 (M_5): C_11 (+1)
    # s=6 (M_6): C_00 (+1)
    
    # Corrected W_factors:
    W_factors = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(8):
        a = t // 4
        b = (t // 2) % 2
        c = t % 2
        
        for s in range(7):
            r = 7 * t + s
            
            # Fill U
            for (i, j_loc, coeff) in A_factors[s]:
                j_glob = 2 * a + j_loc
                u_idx = i * m + j_glob
                U[u_idx][r] += coeff
            
            # Fill V
            for (j_loc, k_loc, coeff) in B_factors[s]:
                j_glob = 2 * a + j_loc
                k_glob = 4 * b + 2 * c + k_loc
                v_idx = j_glob * p + k_glob
                V[v_idx][r] += coeff
            
            # Fill W
            for (i, k_loc, coeff) in W_factors[s]:
                k_glob = 4 * b + 2 * c + k_loc
                w_idx = k_glob * n + i
                W[w_idx][r] += coeff
            
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
