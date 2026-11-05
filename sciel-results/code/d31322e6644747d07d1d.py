"""Block-Strassen rank-189 decomposition for 6x6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    B = 3  # block size
    
    # Strassen 7 products for 2x2 block multiplication
    # Each product: (sum of A-blocks) * (sum of B-blocks)
    # A-blocks: A00, A01, A10, A11 (each 3x3)
    # B-blocks: B00, B01, B10, B11 (each 3x3)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Strassen product, we have a 3x3x3 schoolbook decomposition (9 terms)
    # Total rank = 7 * 9 = 63? No, let me think again.
    
    # Actually, the way to get 189: each of the 7 Strassen "products" is a 3x3 matrix product,
    # which requires 9 rank-1 terms. But the coefficients from Strassen (±1) are applied
    # to the output W matrix. So we have 7*9 = 63 rank-1 terms in U and V, and the W
    # matrix encodes the Strassen combination.
    
    # Wait, I need to be more careful. Let me think of it as:
    # The full 6x6x6 tensor T[i,k,j] = 1 (for the standard matmul tensor where
    # C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Block structure: i = i_b*3 + i_l, j = j_b*3 + j_l, k = k_b*3 + k_l
    # where i_b, j_b, k_b in {0,1} and i_l, j_l, k_l in {0,1,2}
    
    # The tensor decomposes as sum over block positions of (block tensor) ⊗ (scalar tensor)
    # Naively: 8 block positions * 9 scalar terms = 72? No, that's not 216 either.
    
    # Let me just directly construct using the known rank-189 structure.
    # 189 = 7 * 27. Each of the 7 Strassen products contributes 27 terms (3*3*3).
    
    # For Strassen product M_r, the U and V factors are sums of block contributions,
    # and for each (i_l, j_l, k_l) in the 3x3x3 grid, we have one rank-1 term.
    
    # Let me define the Strassen products with their A and B block coefficients:
    # Each product r has: A_coefs[4] (for A00,A01,A10,A11) and B_coefs[4] (for B00,B01,B10,B11)
    
    # M1: A=(1,0,0,1), B=(1,0,0,1)  -> (A00+A11)(B00+B11)
    # M2: A=(0,0,1,1), B=(1,0,0,0)  -> (A10+A11)B00
    # M3: A=(1,0,0,0), B=(0,0,1,-1) -> A00(B10-B11)
    # M4: A=(0,0,0,1), B=(0,1,-1,0) -> A11(B01-B10)
    # M5: A=(1,1,0,0), B=(0,0,0,1)  -> (A00+A01)B11
    # M6: A=(0,0,1,-1), B=(1,0,1,0) -> (A10-A00)(B00+B10)
    # M7: A=(0,1,0,-1), B=(0,1,0,1) -> (A01-A11)(B01+B11)
    
    # Output combination for each C-block (C00,C01,C10,C11):
    # C00: M1+M4-M5+M7
    # C01: M3+M5
    # C10: M2+M4
    # C11: M1-M2+M3+M6
    
    A_coefs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1], # M7
    ]
    B_coefs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [0, 1, -1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 1, 0],  # M6
        [0, 1, 0, 1],  # M7
    ]
    # Output: which M's contribute to which C-block, with what sign
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    # C-blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    out_coefs = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],   # M1, M2, M3, M4, M5, M6, M7
        (0, 1): [0, 0, 1, 0, 1, 0, 0],
        (1, 0): [0, 1, 0, 1, 0, 0, 0],
        (1, 1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for r in range(7):
        for il in range(B):
            for jl in range(B):
                for kl in range(B):
                    # This rank-1 term corresponds to Strassen product r,
                    # scalar positions (il, jl, kl)
                    
                    # U[i, j] where i = i_b*3+il, j = j_b*3+jl
                    # For each A-block (i_b, j_b), coefficient is A_coefs[r][i_b*2+j_b]
                    for ib in range(2):
                        for jb in range(2):
                            a_coef = A_coefs[r][ib * 2 + jb]
                            if a_coef != 0:
                                i = ib * B + il
                                j = jb * B + jl
                                U[i * m + j][col] = a_coef
                    
                    # V[j, k] where j = j_b*3+jl, k = k_b*3+kl
                    for jb in range(2):
                        for kb in range(2):
                            b_coef = B_coefs[r][jb * 2 + kb]
                            if b_coef != 0:
                                j = jb * B + jl
                                k = kb * B + kl
                                V[j * p + k][col] = b_coef
                    
                    # W[k, i] where k = k_b*3+kl, i = i_b*3+il
                    # C-block (i_b, k_b), coefficient from out_coefs
                    for ib in range(2):
                        for kb in range(2):
                            w_coef = out_coefs[(ib, kb)][r]
                            if w_coef != 0:
                                k = kb * B + kl
                                i = ib * B + il
                                W[k * n + i][col] = w_coef
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
