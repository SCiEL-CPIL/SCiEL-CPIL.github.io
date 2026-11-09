"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 3x3 block partitioning with 2x2 blocks
    # Rank = 3*3*3 (blocks) * 7 (Strassen) = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm linear forms for 2x2 matrices
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # Coefficients for U (A part) and V (B part) for each of the 7 multiplications
    # Local indices in 2x2 block: 00, 01, 10, 11
    # Map local (r,c) to flat index 0..3
    # 0:00, 1:01, 2:10, 3:11
    
    strassen_A = [
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [1, 0, 1, 0],  # M6: a10 - a00 (coeffs 1, 0, 1, 0? No, a10 is idx 2, a00 is idx 0. So -1, 0, 1, 0)
        [0, 1, 0, -1]  # M7: a01 - a11 (coeffs 0, 1, 0, -1)
    ]
    # Correction for M6: a10 - a00. a00 is idx 0, a10 is idx 2.
    strassen_A[5] = [-1, 0, 1, 0]
    
    strassen_B = [
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 0, 1, -1], # M3: b10 - b11
        [-1, 1, 0, 0], # M4: b01 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ]
    
    # Reconstruction coefficients for C (Z) from M1..M7
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # W coeffs for local indices 0:00, 1:01, 2:10, 3:11
    # Each entry is a list of 7 coeffs corresponding to M1..M7
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # Z00
        [0, 0, 1, 0, 1, 0, 0],   # Z01
        [0, 1, 0, 1, 0, 0, 0],   # Z10
        [1, -1, 1, 0, 0, 1, 0]   # Z11
    ]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # Update U
                    # A block is at rows 2i, 2i+1 and cols 2k, 2k+1
                    # Global row r, col c -> index r*6 + c
                    # Local (lr, lc) -> global (2i+lr, 2k+lc)
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff = strassen_A[s][local_idx]
                            if coeff != 0:
                                global_row = 2 * i + lr
                                global_col = 2 * k + lc
                                U[global_row * 6 + global_col][col] = coeff
                                
                    # Update V
                    # B block is at rows 2k, 2k+1 and cols 2j, 2j+1
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff = strassen_B[s][local_idx]
                            if coeff != 0:
                                global_row = 2 * k + lr
                                global_col = 2 * j + lc
                                V[global_row * 6 + global_col][col] = coeff
                                
                    # Update W
                    # C block is at rows 2i, 2i+1 and cols 2j, 2j+1
                    # W index convention: W[col * n + row]
                    # Global row r, col c -> index c*6 + r
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff = strassen_W[local_idx][s]
                            if coeff != 0:
                                global_row = 2 * i + lr
                                global_col = 2 * j + lc
                                W[global_col * 6 + global_row][col] = coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
