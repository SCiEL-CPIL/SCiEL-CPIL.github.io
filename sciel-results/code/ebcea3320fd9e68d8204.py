"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm on 2x2 blocks
    # We split the 14 columns of B into 7 pairs.
    # For each pair, we apply Strassen's 7-product algorithm.
    # Total rank = 7 (Strassen) * 7 (blocks) = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 00, 01, 10, 11 -> U indices: 0, 1, 2, 3
    # u_1 = a_00 + a_11
    # u_2 = a_10 + a_11
    # u_3 = a_00
    # u_4 = a_11
    # u_5 = a_00 + a_01
    # u_6 = a_10 - a_00
    # u_7 = a_01 - a_11
    
    a_idx = [(0, 0), (0, 1), (1, 0), (1, 1)] # 0, 1, 2, 3
    strassen_A_forms = [
        [(0, 1), (3, 1)],      # a00 + a11
        [(2, 1), (3, 1)],      # a10 + a11
        [(0, 1)],              # a00
        [(3, 1)],              # a11
        [(0, 1), (1, 1)],      # a00 + a01
        [(2, 1), (0, -1)],     # a10 - a00
        [(1, 1), (3, -1)]      # a01 - a11
    ]
    
    # Strassen reconstruction coefficients for C
    # C_00 = M1 - M4 + M7 - M6
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M3 + M5 - M2
    # M indices 0..6 correspond to the 7 products
    
    # W matrix maps products to output C entries
    # C indices: k*n + i. 
    # For a specific block k, the output columns are 2k and 2k+1.
    # So C_{i, 2k} and C_{i, 2k+1}.
    # W index for C_{i, j} is j*n + i.
    
    col = 0
    for block in range(7):
        # Columns 2*block and 2*block+1
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # B linear forms for this block
        # B indices in V matrix: j*p + k.
        # b_00 -> B_{0, c0} -> V index 0*14 + c0
        # b_01 -> B_{0, c1} -> V index 0*14 + c1
        # b_10 -> B_{1, c0} -> V index 1*14 + c0
        # b_11 -> B_{1, c1} -> V index 1*14 + c1
        
        b_idx = [
            (0 * 14 + c0, 1), (0 * 14 + c1, 1), # b00, b01
            (1 * 14 + c0, 1), (1 * 14 + c1, 1)  # b10, b11
        ]
        # Wait, Strassen forms:
        # v_1 = b00 + b11
        # v_2 = b00
        # v_3 = b01 - b11
        # v_4 = b00 - b10
        # v_5 = b11
        # v_6 = b00 + b01
        # v_7 = b10 + b11
        
        strassen_B_forms = [
            [(0 * 14 + c0, 1), (1 * 14 + c1, 1)],      # b00 + b11
            [(0 * 14 + c0, 1)],                         # b00
            [(0 * 14 + c1, 1), (1 * 14 + c1, -1)],      # b01 - b11
            [(0 * 14 + c0, 1), (1 * 14 + c0, -1)],      # b00 - b10
            [(1 * 14 + c1, 1)],                         # b11
            [(0 * 14 + c0, 1), (0 * 14 + c1, 1)],       # b00 + b01
            [(1 * 14 + c0, 1), (1 * 14 + c1, 1)]        # b10 + b11
        ]
        
        for r in range(7):
            # Fill U
            for (idx, coeff) in strassen_A_forms[r]:
                U[idx][col] += coeff
            
            # Fill V
            for (idx, coeff) in strassen_B_forms[r]:
                V[idx][col] += coeff
            
            # Fill W
            # C_00 (i=0, j=c0) gets M1 - M4 + M7 - M6
            # C_01 (i=0, j=c1) gets M3 + M5
            # C_10 (i=1, j=c0) gets M2 + M4
            # C_11 (i=1, j=c1) gets M1 - M3 + M5 - M2
            
            # W index for C_{i, j} is j*n + i = j*2 + i
            w_00 = c0 * 2 + 0
            w_01 = c1 * 2 + 0
            w_10 = c0 * 2 + 1
            w_11 = c1 * 2 + 1
            
            if r == 0: # M1
                W[w_00][col] += 1
                W[w_11][col] += 1
            elif r == 1: # M2
                W[w_10][col] += 1
                W[w_11][col] -= 1
            elif r == 2: # M3
                W[w_01][col] += 1
                W[w_11][col] -= 1
            elif r == 3: # M4
                W[w_00][col] -= 1
                W[w_10][col] += 1
            elif r == 4: # M5
                W[w_01][col] += 1
                W[w_11][col] += 1
            elif r == 5: # M6
                W[w_00][col] -= 1
            elif r == 6: # M7
                W[w_00][col] += 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
