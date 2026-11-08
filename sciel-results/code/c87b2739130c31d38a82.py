"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for a 2x2 * 2x2 multiplication
    # Q (U part) indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # R (V part) indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P (W part) indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # s0: (q00+q11)(r00+r11)
    # s1: (q10+q11)r00
    # s2: q00(r01-r11)
    # s3: q11(r10-r00)
    # s4: (q00+q01)r11
    # s5: (q10-q00)(r00+r01)
    # s6: (q01-q11)(r10+r11)
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1}, # s0
        {(1, 0): 1, (1, 1): 1}, # s1
        {(0, 0): 1},            # s2
        {(1, 1): 1},            # s3
        {(0, 0): 1, (0, 1): 1}, # s4
        {(1, 0): 1, (0, 0): -1},# s5
        {(0, 1): 1, (1, 1): -1} # s6
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1}, # s0
        {(0, 0): 1},            # s1
        {(0, 1): 1, (1, 1): -1},# s2
        {(1, 0): 1, (0, 0): -1},# s3
        {(1, 1): 1},            # s4
        {(0, 0): 1, (0, 1): 1}, # s5
        {(1, 0): 1, (1, 1): 1}  # s6
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # s0 -> p00(+1), p11(+1)
        {(1, 0): 1, (1, 1): -1}, # s1 -> p10(+1), p11(-1)
        {(0, 1): 1, (1, 1): 1},  # s2 -> p01(+1), p11(+1)
        {(0, 0): 1, (1, 0): 1},  # s3 -> p00(+1), p10(+1)
        {(0, 0): -1, (0, 1): 1}, # s4 -> p00(-1), p01(+1)
        {(1, 1): 1},             # s5 -> p11(+1)
        {(0, 0): 1}              # s6 -> p00(+1)
    ]
    
    col = 0
    # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3)
    # B is 4x6. Split into B0 (rows 0,1) and B1 (rows 2,3)
    # For each block pair (A_block, B_block), we do 2x2 * 2x6
    # 2x2 * 2x6 is split into 3 * (2x2 * 2x2)
    
    for a_block in range(2): # 0: cols 0-1, 1: cols 2-3
        for b_block in range(2): # 0: rows 0-1, 1: rows 2-3
            for k_block in range(3): # 0: cols 0-1, 1: cols 2-3, 2: cols 4-5
                for s in range(7):
                    # Update U (A part)
                    # A index: i * m + j
                    # j is in [a_block*2, a_block*2+1]
                    for (i_loc, j_loc), coeff in U_coeffs[s].items():
                        i = i_loc
                        j = a_block * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    
                    # Update V (B part)
                    # B index: j * p + k
                    # j is in [b_block*2, b_block*2+1]
                    # k is in [k_block*2, k_block*2+1]
                    for (j_loc, k_loc), coeff in V_coeffs[s].items():
                        j = b_block * 2 + j_loc
                        k = k_block * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    
                    # Update W (C part)
                    # C index: k * n + i
                    # k is in [k_block*2, k_block*2+1]
                    for (i_loc, k_loc), coeff in W_coeffs[s].items():
                        i = i_loc
                        k = k_block * 2 + k_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
