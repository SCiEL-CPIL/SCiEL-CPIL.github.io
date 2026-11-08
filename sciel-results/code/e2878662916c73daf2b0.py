import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strategy: 2x2 blocks of 3x3 submatrices
    # Block multiplication using 7-term Strassen
    # Each block product (3x3 x 3x3) using 27-term naive decomposition
    # Total rank = 7 * 27 = 189
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 7 * 27 # 8 * 7 * 27? No.
    # Wait, the tiling is 2x2 blocks.
    # The Strassen algorithm applies to the 2x2 block structure.
    # The "rank" of the 2x2 block multiplication is 7.
    # The "rank" of the 3x3 sub-block multiplication is 27.
    # Total rank is 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen patterns for 2x2 blocks
    # U_s, V_s, W_s are 2x2 matrices of coefficients
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    col_idx = 0
    for s in range(7):
        u_block, v_block, w_block = strassen_patterns[s]
        # Naive decomposition of 3x3 multiplication: 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # This term corresponds to A_virt[i,j] * B_virt[j,k] -> C_virt[i,k]
                    
                    # Construct U vector for this term
                    # A_virt[i,j] = sum_{a,b} U_s[a,b] A_block(a,b)[i,j]
                    for (a, b), val in u_block.items():
                        r = a * block_size + i
                        c = b * block_size + j
                        U[r * m + c][col_idx] += val
                    
                    # Construct V vector for this term
                    # B_virt[j,k] = sum_{c,d} V_s[c,d] B_block(c,d)[j,k]
                    for (c, d), val in v_block.items():
                        r = c * block_size + j
                        col = d * block_size + k
                        V[r * p + col][col_idx] += val
                    
                    # Construct W vector for this term
                    # C_block(I,K)[i,k] += W_s[I,K] * C_virt[i,k]
                    for (I, K), val in w_block.items():
                        r = K * block_size + k
                        c = I * block_size + i
                        W[r * n + c][col_idx] += val
                    
                    col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
