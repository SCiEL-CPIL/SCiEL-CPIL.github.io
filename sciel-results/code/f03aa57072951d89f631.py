"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 98 via recursive Strassen
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    base_terms = strassen_2x2_terms()
    
    # Maps for A, B, W blocks
    # Block IDs: 0=11, 1=12, 2=21, 3=22
    A_maps = [
        [(0, 1), (3, 1)],      # M1: A11 + A22
        [(2, 1), (3, 1)],      # M2: A21 + A22
        [(0, 1)],              # M3: A11
        [(3, 1)],              # M4: A22
        [(0, 1), (1, 1)],      # M5: A11 + A12
        [(2, 1), (0, -1)],     # M6: A21 - A11
        [(1, 1), (3, -1)]      # M7: A12 - A22
    ]
    
    B_maps = [
        [(0, 1), (3, 1)],      # M1: B11 + B22
        [(0, 1)],              # M2: B11
        [(1, 1), (3, -1)],     # M3: B12 - B22
        [(2, 1), (0, -1)],     # M4: B21 - B11
        [(3, 1)],              # M5: B22
        [(0, 1), (1, 1)],      # M6: B11 + B12
        [(2, 1), (3, 1)]       # M7: B21 + B22
    ]
    
    # W_maps[s][side] where side 0=L, 1=R
    # C Block IDs: 0=11, 1=12, 2=21, 3=22
    W_maps = [
        [[(0, 1), (3, 1)], [(0, 1), (3, 1)]],   # M1 -> C11, C22
        [[(0, -1), (2, 1)], [(0, -1), (2, 1)]], # M2 -> C11(-), C21
        [[(1, 1), (3, -1)], [(1, 1), (3, -1)]], # M3 -> C12, C22(-)
        [[(0, 1), (2, 1)], [(0, 1), (2, 1)]],   # M4 -> C11, C21
        [[(1, 1), (3, 1)], [(1, 1), (3, 1)]],   # M5 -> C12, C22
        [[(0, 1)], [(0, 1)]],                   # M6 -> C11
        [[(3, 1)], [(3, 1)]]                    # M7 -> C22
    ]
    
    term_idx = 0
    for s in range(7):
        for side in range(2): # 0 for L, 1 for R
            for u, v, w in base_terms:
                # Process U
                for l in range(4):
                    val = u[l]
                    if val == 0: continue
                    r_loc = l // 2
                    c_loc = l % 2
                    for block_id, coeff in A_maps[s]:
                        r_off = (block_id // 2) * 2
                        c_off = (block_id % 2) * 2
                        g_r = r_off + r_loc
                        g_c = c_off + c_loc
                        g_idx = g_r * 4 + g_c
                        U[g_idx][term_idx] += val * coeff
                
                # Process V
                for l in range(4):
                    val = v[l]
                    if val == 0: continue
                    r_loc = l // 2
                    c_loc = l % 2
                    for block_id, coeff in B_maps[s]:
                        r_off = (block_id // 2) * 2
                        c_off = (block_id % 2) * 4
                        side_off = 0 if side == 0 else 2
                        g_r = r_off + r_loc
                        g_c = c_off + side_off + c_loc
                        g_idx = g_r * 8 + g_c
                        V[g_idx][term_idx] += val * coeff
                
                # Process W
                for l in range(4):
                    val = w[l]
                    if val == 0: continue
                    r_loc = l // 2
                    c_loc = l % 2
                    for block_id, coeff in W_maps[s][side]:
                        r_off = (block_id // 2) * 2
                        c_off = (block_id % 2) * 4
                        side_off = 0 if side == 0 else 2
                        g_r = r_off + r_loc
                        g_c = c_off + side_off + c_loc
                        # W index is k*n + i
                        g_idx = g_c * 4 + g_r
                        W[g_idx][term_idx] += val * coeff
                
                term_idx += 1
                
    return {"U": U, "V": V, "W": W}

def strassen_2x2_terms():
    """Return 7 (u, v, w) terms for 2x2 x 2x2 multiplication."""
    terms = []
    # M1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)e
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    # M3 = a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # M4 = d(g-e)
    terms.append(([0,0,0,1], [-1,0,1,0], [1,1,0,0]))
    # M5 = (a+b)h
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6 = (c-a)(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
    return terms

# EVOLVE-BLOCK-END
