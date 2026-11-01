"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49 using Strassen for each 2x2 block of B
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen coefficients for A (U)
    # M1: a00+a11, M2: a10+a11, M3: a00, M4: a11, M5: a00+a01, M6: a10-a00, M7: a01-a11
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # For each block k in 0..6
    for k in range(7):
        # Indices in V for B_k
        # B is 2x14. Block k uses cols 2k, 2k+1.
        # V index: row * p + col
        v_b00 = 0 * p + 2 * k
        v_b01 = 0 * p + 2 * k + 1
        v_b10 = 1 * p + 2 * k
        v_b11 = 1 * p + 2 * k + 1
        
        # V coefficients for B_k
        # M1: b00+b11, M2: b00, M3: b01-b11, M4: b10-b00, M5: b11, M6: b00+b01, M7: b10+b11
        V_coeffs = [
            {v_b00: 1, v_b11: 1},
            {v_b00: 1},
            {v_b01: 1, v_b11: -1},
            {v_b10: 1, v_b00: -1},
            {v_b11: 1},
            {v_b00: 1, v_b01: 1},
            {v_b10: 1, v_b11: 1}
        ]
        
        # Indices in W for C_k
        # W index: col * n + row
        # C_k is 2x2 block at cols 2k, 2k+1
        # c00 -> col 2k, row 0 -> 2k * 2 + 0
        w_c00 = 2 * k * n + 0
        w_c10 = 2 * k * n + 1
        w_c01 = (2 * k + 1) * n + 0
        w_c11 = (2 * k + 1) * n + 1
        
        # W coefficients for C_k
        # M1: c00+1, c11+1
        # M2: c10+1, c11-1
        # M3: c01+1, c11+1
        # M4: c00+1, c10+1
        # M5: c00-1, c01+1
        # M6: c11+1
        # M7: c00+1
        W_coeffs = [
            {w_c00: 1, w_c11: 1},
            {w_c10: 1, w_c11: -1},
            {w_c01: 1, w_c11: 1},
            {w_c00: 1, w_c10: 1},
            {w_c00: -1, w_c01: 1},
            {w_c11: 1},
            {w_c00: 1}
        ]
        
        for j in range(7):
            col = k * 7 + j
            
            # Fill U
            for i in range(4):
                U[i][col] = U_coeffs[j][i]
                
            # Fill V
            for idx, val in V_coeffs[j].items():
                V[idx][col] += val
                
            # Fill W
            for idx, val in W_coeffs[j].items():
                W[idx][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
