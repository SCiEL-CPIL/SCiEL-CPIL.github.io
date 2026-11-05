"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Block-Strassen
    # 3x3 blocks of 2x2 matrices
    # Each 2x2 block product uses Strassen's 7-term algorithm
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # P0 = (a00+a11)(b00+b11)
    # P1 = (a10+a11)b00
    # P2 = a00(b10-b11)
    # P3 = a11(b01-b00)
    # P4 = (a00+a10)b11
    # P5 = (a11-a00)(b00+b10)
    # P6 = (a01-a11)(b11+b01)
    
    # Coefficients for U (A), V (B), W (C)
    # Local indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # U coeffs for each P
    u_coeffs = [
        {0: 1, 3: 1},       # P0: a00, a11
        {2: 1, 3: 1},       # P1: a10, a11
        {0: 1},             # P2: a00
        {3: 1},             # P3: a11
        {0: 1, 2: 1},       # P4: a00, a10
        {3: 1, 0: -1},      # P5: a11, -a00
        {1: 1, 3: -1}       # P6: a01, -a11
    ]
    
    # V coeffs for each P
    v_coeffs = [
        {0: 1, 3: 1},       # P0: b00, b11
        {0: 1},             # P1: b00
        {2: 1, 3: -1},      # P2: b10, -b11
        {1: 1, 0: -1},      # P3: b01, -b00
        {3: 1},             # P4: b11
        {0: 1, 2: 1},       # P5: b00, b10
        {3: 1, 1: 1}        # P6: b11, b01
    ]
    
    # W coeffs for each P
    # c00 = P0 + P3 - P4 + P6
    # c01 = P2 + P4
    # c10 = P1 + P3
    # c11 = P0 - P1 + P2 + P5
    w_coeffs = [
        {0: 1, 3: 1},       # P0: c00, c11
        {2: 1, 3: -1},      # P1: c10, -c11
        {1: 1, 3: 1},       # P2: c01, c11
        {0: 1, 2: 1},       # P3: c00, c10
        {0: -1, 1: 1},      # P4: -c00, c01
        {3: 1},             # P5: c11
        {0: 1}              # P6: c00
    ]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for s in range(7):
                    # Global indices for A
                    # a_rc -> row 2i+r, col 2j+c
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            coeff = u_coeffs[s].get(local_idx, 0)
                            if coeff != 0:
                                global_row = (2 * i + r) * m + (2 * j + c)
                                U[global_row][col] += coeff
                            
                    # Global indices for B
                    # b_cd -> row 2j+c, col 2k+d
                    for c in range(2):
                        for d in range(2):
                            local_idx = c * 2 + d
                            coeff = v_coeffs[s].get(local_idx, 0)
                            if coeff != 0:
                                global_row = (2 * j + c) * p + (2 * k + d)
                                V[global_row][col] += coeff
                                
                    # Global indices for C
                    # c_rd -> row 2i+r, col 2k+d
                    # W index: k*n + i -> (2k+d)*n + (2i+r)
                    for r in range(2):
                        for d in range(2):
                            local_idx = r * 2 + d
                            coeff = w_coeffs[s].get(local_idx, 0)
                            if coeff != 0:
                                global_row = (2 * k + d) * n + (2 * i + r)
                                W[global_row][col] += coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
