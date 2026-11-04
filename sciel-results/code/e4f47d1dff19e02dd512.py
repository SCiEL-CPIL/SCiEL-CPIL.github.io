"""Rewritten program using a block-Strassen construction for rank-minimal search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen local matrices (4x7)
    # U_local: rows 0..3 correspond to X00, X01, X10, X11
    U_local = [
        [1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, -1, 0, 0, -1, 1]
    ]
    # V_local: rows 0..3 correspond to Y00, Y01, Y10, Y11
    V_local = [
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1]
    ]
    # W_local: rows 0..3 correspond to Z00, Z01, Z10, Z11
    W_local = [
        [1, 0, 1, -1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 0, 1, 1, 0]
    ]
    
    for g in range(3):
        for m_idx in range(4):
            # Determine X and Y block structures
            if m_idx == 0: # M1: X=A00, Y=B0g
                x_blocks = [(0, 0, 1)]
                y_row_block = 0
            elif m_idx == 1: # M2: X=A01, Y=B1g
                x_blocks = [(0, 1, 1)]
                y_row_block = 1
            elif m_idx == 2: # M3: X=A00+A10, Y=B0g
                x_blocks = [(0, 0, 1), (1, 0, 1)]
                y_row_block = 0
            else: # M4: X=A01+A11, Y=B1g
                x_blocks = [(0, 1, 1), (1, 1, 1)]
                y_row_block = 1
            
            # Process 7 Strassen terms
            for k in range(7):
                global_col = col + k
                
                # U
                for local_i in range(4):
                    u_val = U_local[local_i][k]
                    if u_val != 0:
                        li_r = local_i // 2
                        li_c = local_i % 2
                        for (rb, cb, coeff) in x_blocks:
                            gr = rb * 2 + li_r
                            gc = cb * 2 + li_c
                            idx = gr * m + gc
                            U[idx][global_col] += u_val * coeff
                
                # V
                for local_j in range(4):
                    v_val = V_local[local_j][k]
                    if v_val != 0:
                        lj_r = local_j // 2
                        lj_c = local_j % 2
                        gr = y_row_block * 2 + lj_r
                        gc = g * 2 + lj_c
                        idx = gr * p + gc
                        V[idx][global_col] += v_val
                
                # W
                for local_z in range(4):
                    w_val = W_local[local_z][k]
                    if w_val != 0:
                        lz_r = local_z // 2
                        lz_c = local_z % 2
                        
                        # Determine coefficient for C
                        if lz_r == 0:
                            coeff_c = 1 if m_idx in [0, 1] else 0
                        else:
                            coeff_c = -1 if m_idx in [0, 1] else 1
                        
                        if coeff_c != 0:
                            gr = lz_r * 2 + lz_c
                            gc = g * 2 + lz_c
                            idx = gc * n + gr
                            W[idx][global_col] += w_val * coeff_c
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
