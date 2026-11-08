import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen for 2x2 blocks
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for block level (7 steps)
    # U_S, V_S, W_S are dicts: block_index -> coeff
    # Block indices: 0: TL, 1: TR, 2: BL, 3: BR
    strassen_blocks = [
        # S=1
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        # S=2
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: 1}},
        # S=3
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        # S=4
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        # S=5
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        # S=6
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        # S=7
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}},
    ]
    
    # Define Strassen coefficients for 2x2 matrix multiplication (7 products)
    # Each product is (lin_form_X) * (lin_form_Y)
    # X entries: 00, 01, 10, 11
    # Y entries: 00, 01, 10, 11
    # Result Z entries: 00, 01, 10, 11
    # p_s contributes to Z with coeffs w_s
    strassen_2x2 = [
        # s=1: (x00+x11)(y00+y11) -> z00+=1, z11+=1
        {'ux': {0: 1, 3: 1}, 'vy': {0: 1, 3: 1}, 'wz': {0: 1, 3: 1}},
        # s=2: (x10+x11)y00 -> z10+=1, z11+=1
        {'ux': {2: 1, 3: 1}, 'vy': {0: 1}, 'wz': {2: 1, 3: 1}},
        # s=3: x00(y01-y11) -> z01+=1, z11+=1
        {'ux': {0: 1}, 'vy': {1: 1, 3: -1}, 'wz': {1: 1, 3: 1}},
        # s=4: x11(y10-y00) -> z00+=1, z10+=1
        {'ux': {3: 1}, 'vy': {2: 1, 0: -1}, 'wz': {0: 1, 2: 1}},
        # s=5: (x00+x01)y11 -> z00-=1, z01+=1
        {'ux': {0: 1, 1: 1}, 'vy': {3: 1}, 'wz': {0: -1, 1: 1}},
        # s=6: (x10-x00)(y00+y01) -> z11+=1
        {'ux': {2: 1, 0: -1}, 'vy': {0: 1, 1: 1}, 'wz': {3: 1}},
        # s=7: (x01-x11)(y10+y11) -> z00+=1
        {'ux': {1: 1, 3: -1}, 'vy': {2: 1, 3: 1}, 'wz': {0: 1}},
    ]
    
    # Helper to get block row/col offsets
    # Block 0: r=0, c=0
    # Block 1: r=0, c=2
    # Block 2: r=2, c=0
    # Block 3: r=2, c=2
    block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    col = 0
    for S_idx, S in enumerate(strassen_blocks):
        U_S = S['U']
        V_S = S['V']
        W_S = S['W']
        
        for s_idx, s in enumerate(strassen_2x2):
            ux = s['ux']
            vy = s['vy']
            wz = s['wz']
            
            # Build U vector for this column
            # U[global_idx] = sum over block b, local i,j of ux[i,j] * U_S[b] * delta(global == 2*row_b + i, 2*col_b + j)
            for i in range(2):
                for j in range(2):
                    if (i * 2 + j) in ux: # Wait, ux keys are 0,1,2,3 corresponding to 00,01,10,11?
                        # My ux dict uses 0:00, 1:01, 2:10, 3:11
                        pass
                    
            # Let's use explicit mapping
            # ux keys: 0 (00), 1 (01), 2 (10), 3 (11)
            for x_idx, coeff_x in ux.items():
                if coeff_x == 0: continue
                i_loc = x_idx // 2
                j_loc = x_idx % 2
                
                for b, coeff_b in U_S.items():
                    if coeff_b == 0: continue
                    r_b, c_b = block_offsets[b]
                    global_i = r_b + i_loc
                    global_j = c_b + j_loc
                    U[global_i * 4 + global_j][col] += coeff_x * coeff_b
            
            # Build V vector for this column
            for y_idx, coeff_y in vy.items():
                if coeff_y == 0: continue
                i_loc = y_idx // 2
                j_loc = y_idx % 2
                
                for b, coeff_b in V_S.items():
                    if coeff_b == 0: continue
                    r_b, c_b = block_offsets[b]
                    global_i = r_b + i_loc
                    global_j = c_b + j_loc
                    V[global_i * 4 + global_j][col] += coeff_y * coeff_b
            
            # Build W vector for this column
            # W[global_idx] = sum over block b, local i,j of wz[i,j] * W_S[b] * delta(global == 2*row_b + i, 2*col_b + j)
            # Note: W index is k*n + i (row k, col i of C)
            # C is p x n = 4 x 4.
            # Block b of C corresponds to rows 2*r_b..2*r_b+1, cols 2*c_b..2*c_b+1.
            for z_idx, coeff_z in wz.items():
                if coeff_z == 0: continue
                i_loc = z_idx // 2
                j_loc = z_idx % 2
                
                for b, coeff_b in W_S.items():
                    if coeff_b == 0: continue
                    r_b, c_b = block_offsets[b]
                    global_i = r_b + i_loc
                    global_j = c_b + j_loc
                    # W index: global_i * 4 + global_j
                    W[global_i * 4 + global_j][col] += coeff_z * coeff_b
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
