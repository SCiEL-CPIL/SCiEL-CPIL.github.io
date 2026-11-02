import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # Strassen-style decomposition for 6x6x6 using 2x2 blocks of 3x3 matrices.
    # We view the 6x6 matrix as a 2x2 block matrix of 3x3 blocks.
    # Block indices: (b_i, b_j) where b_i in {0,1} is row-block, b_j in {0,1} is col-block.
    # Local indices: (i0, j0) in {0,1,2}x{0,1,2}.
    # Global row index i = 3*b_i + i0
    # Global col index j = 3*b_j + j0
    # Global k index k = 3*b_k + k0

    # Strassen's 7 products for 2x2 block multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each product is a 3x3x3 multiplication tensor, which has rank 27 (schoolbook).
    # So total rank = 7 * 27 = 189.

    rank = 7 * 27
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189

    # For each of the 7 Strassen products r in 0..6:
    #   For each local 3x3x3 schoolbook term (i0, j0, k0) in 0..2:
    #     c = r * 27 + (i0 * 9 + j0 * 3 + k0)  -- wait, need to be careful with indexing.
    #     Actually, for a fixed r, the 27 terms correspond to (i0, j0, k0) in {0,1,2}^3.
    #     U[i, j, c] = alpha_{r, i0, j0, i, j}
    #     V[j, k, c] = beta_{r, j0, k0, j, k}
    #     W[k, i, c] = w_{r, i0, j0, k0, i, k}  -- but W is indexed by (k, i)

    # Let's define the Strassen coefficients.
    # For product r, A^{(r)} = sum over block pairs with coefficients.
    # B^{(r)} = sum over block pairs with coefficients.
    #
    # r=0: A^(0) = A00 + A11, B^(0) = B00 + B11
    # r=1: A^(1) = A10 + A11, B^(1) = B00
    # r=2: A^(2) = A00,       B^(2) = B01 - B11
    # r=3: A^(3) = A11,       B^(3) = B10 - B00
    # r=4: A^(4) = A00 + A01, B^(4) = B11
    # r=5: A^(5) = A10 - A00, B^(5) = B00 + B01
    # r=6: A^(6) = A01 - A11, B^(6) = B10 + B11

    # A_block_coeffs[r][bi][bj] = coefficient for A[bi, bj]
    A_block_coeffs = [
        {(0,0): 1, (1,1): 1},      # r=0
        {(1,0): 1, (1,1): 1},      # r=1
        {(0,0): 1},                # r=2
        {(1,1): 1},                # r=3
        {(0,0): 1, (0,1): 1},      # r=4
        {(1,0): 1, (0,0): -1},     # r=5
        {(0,1): 1, (1,1): -1},     # r=6
    ]

    # B_block_coeffs[r][bk][bj] = coefficient for B[bj, bk]  (B is indexed by [row, col], so B[j,k] with j=row, k=col)
    # Wait, B is m x p = 6x6. Block B[bj][bk] where bj is row-block, bk is col-block.
    B_block_coeffs = [
        {(0,0): 1, (1,1): 1},      # r=0: B00 + B11
        {(0,0): 1},                # r=1: B00
        {(0,1): 1, (1,1): -1},     # r=2: B01 - B11
        {(1,0): 1, (0,0): -1},     # r=3: B10 - B00
        {(1,1): 1},                # r=4: B11
        {(0,0): 1, (0,1): 1},      # r=5: B00 + B01
        {(1,0): 1, (1,1): 1},      # r=6: B10 + B11
    ]

    # C_block_coeffs[r][bi][bk] = coefficient for C[bi, bk]
    C_block_coeffs = [
        {(0,0): 1, (1,1): 1},      # r=0: M1 contributes to C00 and C11
        {(1,0): 1},                # r=1: M2 contributes to C10
        {(0,1): 1},                # r=2: M3 contributes to C01
        {(0,0): 1, (1,0): 1},      # r=3: M4 contributes to C00 and C10
        {(0,0): -1},               # r=4: M5 contributes -1 to C00
        {(1,1): 1},                # r=5: M6 contributes to C11
        {(0,0): 1, (0,1): 1},      # r=6: M7 contributes to C00 and C01
    ]

    # Wait, let me re-verify the C coefficients:
    # C00 = M1 + M4 - M5 + M7  => r0:+1, r3:+1, r4:-1, r6:+1
    # C01 = M3 + M5            => r2:+1, r4:+1
    # C10 = M2 + M4            => r1:+1, r3:+1
    # C11 = M1 - M2 + M3 + M6  => r0:+1, r1:-1, r2:+1, r5:+1

    C_block_coeffs = [
        {(0,0): 1, (1,1): 1},      # r=0
        {(1,0): 1, (1,1): -1},     # r=1
        {(0,1): 1, (1,1): 1},      # r=2
        {(0,0): 1, (1,0): 1},      # r=3
        {(0,0): -1, (0,1): 1},     # r=4
        {(1,1): 1},                # r=5
        {(0,0): 1, (0,1): 1},      # r=6
    ]

    for r in range(7):
        for i0 in range(3):
            for j0 in range(3):
                for k0 in range(3):
                    c = r * 27 + (i0 * 9 + j0 * 3 + k0)

                    # U[i, j, c]: i = 3*bi + i0, j = 3*bj + j0
                    # For A^(r) = sum_{(bi,bj)} coeff_A * A[bi,bj]
                    # U[i,j,c] = coeff_A for the block (bi,bj) where i = 3*bi+i0, j = 3*bj+j0
                    for bi in range(2):
                        for bj in range(2):
                            coeff = A_block_coeffs[r].get((bi, bj), 0)
                            if coeff != 0:
                                i = 3 * bi + i0
                                j = 3 * bj + j0
                                U[i * m + j][c] += coeff

                    # V[j, k, c]: j = 3*bj + j0, k = 3*bk + k0
                    # For B^(r) = sum_{(bj,bk)} coeff_B * B[bj,bk]
                    for bj in range(2):
                        for bk in range(2):
                            coeff = B_block_coeffs[r].get((bj, bk), 0)
                            if coeff != 0:
                                j = 3 * bj + j0
                                k = 3 * bk + k0
                                V[j * p + k][c] += coeff

                    # W[k, i, c]: k = 3*bk + k0, i = 3*bi + i0
                    # C[bi,bk] gets contribution from M_r with coeff_C
                    # W[k,i,c] = coeff_C for the block (bi,bk) where k = 3*bk+k0, i = 3*bi+i0
                    for bi in range(2):
                        for bk in range(2):
                            coeff = C_block_coeffs[r].get((bi, bk), 0)
                            if coeff != 0:
                                i = 3 * bi + i0
                                k = 3 * bk + k0
                                W[k * n + i][c] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
