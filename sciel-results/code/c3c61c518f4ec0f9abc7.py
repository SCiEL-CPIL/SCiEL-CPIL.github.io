"""Matrix multiplication 2x2 by 2x10 rank-35 Strassen-based decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 10
    rank = 35  # 5 blocks * 7 (Strassen)
    
    # Build U, V, W as 2D lists of zeros
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # For each of the 5 column-pair blocks (columns 2l, 2l+1 of the 2x10 matrix)
    # we use 7 intermediate products.
    #
    # Let A be the 2x2 left matrix with entries a[i][j] indexed by U row = i*m + j = 2i+j
    # Let B_block be the 2x2 sub-block of the right matrix with entries b[j][k'] 
    #   where k' in {0,1} is the local column, global col = 2l + k'
    #   V row = j*p + (2l + k') = 20j + 2l + k'
    # Output C_block[i][k'] has global output index W row = k*n + i = 2k + i where k = 2l + k'
    #   W row = 2*(2l + k') + i = 4l + 2k' + i
    #
    # Strassen's 7 products:
    #   M1 = (a00 + a11) * (b00 + b11)
    #   M2 = (a10 + a11) * b00
    #   M3 = a00 * (b01 - b11)
    #   M4 = a11 * (b10 - b00)
    #   M5 = (a00 + a01) * b11
    #   M6 = (a10 - a00) * (b00 + b01)
    #   M7 = (a01 - a11) * (b10 + b11)
    #
    # Output:
    #   c00 = M1 + M4 - M5 + M7
    #   c01 = M3 + M5
    #   c10 = M2 + M4
    #   c11 = M1 - M2 + M3 + M6
    #
    # U coefficients (a side): for product s, which entries of A appear and with what sign
    # V coefficients (b side): for product s, which entries of B_block appear and with what sign
    # W coefficients (c side): for product s, which entries of C_block get +1 or -1
    
    # U_coeffs[s][i][j] = coefficient of a[i][j] in product s
    # V_coeffs[s][j][k'] = coefficient of b[j][k'] in product s
    # W_coeffs[s][i][k'] = coefficient of c[i][k'] contributed by product s
    
    U_c = [
        # M1: (a00 + a11)
        [[1, 0], [0, 1]],
        # M2: (a10 + a11)
        [[0, 0], [1, 1]],
        # M3: a00
        [[1, 0], [0, 0]],
        # M4: a11
        [[0, 0], [0, 1]],
        # M5: (a00 + a01)
        [[1, 1], [0, 0]],
        # M6: (a10 - a00)
        [[-1, 0], [1, 0]],
        # M7: (a01 - a11)
        [[0, 1], [0, -1]],
    ]
    
    V_c = [
        # M1: (b00 + b11)
        [[1, 0], [0, 1]],
        # M2: b00
        [[1, 0], [0, 0]],
        # M3: (b01 - b11)
        [[0, 1], [0, -1]],
        # M4: (b10 - b00)
        [[-1, 0], [1, 0]],
        # M5: b11
        [[0, 0], [0, 1]],
        # M6: (b00 + b01)
        [[1, 1], [0, 0]],
        # M7: (b10 + b11)
        [[0, 0], [1, 1]],
    ]
    
    W_c = [
        # M1: c00 += 1, c11 += 1
        [[1, 0], [0, 1]],
        # M2: c10 += 1, c11 -= 1
        [[0, 0], [1, -1]],
        # M3: c01 += 1, c11 += 1
        [[0, 1], [0, 1]],
        # M4: c00 += 1, c10 += 1
        [[1, 0], [1, 0]],
        # M5: c00 -= 1, c01 += 1
        [[-1, 1], [0, 0]],
        # M6: c11 += 1
        [[0, 0], [0, 1]],
        # M7: c00 += 1
        [[1, 0], [0, 0]],
    ]
    
    for l in range(5):  # 5 column-pair blocks
        for s in range(7):  # 7 Strassen products per block
            r = l * 7 + s  # global rank index
            
            # Fill U column r
            for i in range(2):
                for j in range(2):
                    coeff = U_c[s][i][j]
                    if coeff != 0:
                        U[2 * i + j][r] = coeff
            
            # Fill V column r
            for j in range(2):
                for kp in range(2):  # local column k'
                    coeff = V_c[s][j][kp]
                    if coeff != 0:
                        global_col = 2 * l + kp
                        V[j * 20 + global_col][r] = coeff
            
            # Fill W column r
            for i in range(2):
                for kp in range(2):  # local column k'
                    coeff = W_c[s][i][kp]
                    if coeff != 0:
                        global_col = 2 * l + kp
                        W[2 * global_col + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
