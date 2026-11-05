"""Tensor decomposition for 6x6 matrix multiplication using Strassen block structure."""

import time

# Deadline guard: kill after 140s to stay safely under the 150s limit.
DEADLINE = time.time() + 140.0

DIMENSIONS = [6, 6, 6]

# Strassen's 7 multiplications for 2x2 blocks of size 3x3
# Each block is a 3x3 matrix, so each "multiplication" of two 3x3 blocks
# requires rank 27 (schoolbook) in the tensor decomposition.
# Total rank = 7 * 27 = 189.

# Strassen formulas (standard):
# M1 = (A00 + A11) * (B00 + B11)
# M2 = (A10 + A11) * B00
# M3 = A00 * (B01 - B11)
# M4 = A11 * (B10 - B00)
# M5 = (A00 + A01) * B11
# M6 = (A10 - A00) * (B00 + B01)
# M7 = (A01 - A11) * (B10 + B11)
#
# C00 = M1 - M2 + M4 + M7
# C01 = M3 + M5
# C10 = M6 + M4
# C11 = M1 - M3 + M5 - M2


def _build_strassen_block_decomposition():
    """
    Build the factor matrices for 6x6 matrix multiplication using Strassen's
    algorithm on 2x2 blocks of 3x3 matrices.
    
    The 6x6 matrix is partitioned into four 3x3 blocks:
        A = [[A00, A01],
             [A10, A11]]
        B = [[B00, B01],
             [B10, B11]]
        C = [[C00, C01],
             [C10, C11]]
    
    Each 3x3 block multiplication A_ij * B_jk has rank 27 (schoolbook).
    Strassen reduces the 4 block multiplications to 7.
    Total rank = 7 * 27 = 189.
    
    Indexing convention:
    - U has shape (n*m, rank) = (36, 189): row index = i*6 + j (i is row block, j is col within block)
    - V has shape (m*p, rank) = (36, 189): row index = j*6 + k (j is row within block, k is col block)
    - W has shape (p*n, rank) = (36, 189): row index = k*6 + i (k is col block, i is row within block)
      (official trace convention: W[k*n + i, r])
    
    For each Strassen multiplication r (0..6), we have 27 schoolbook terms.
    For Strassen step r, the A-factor is a linear combination of A-blocks,
    and the B-factor is a linear combination of B-blocks.
    The C-contribution is also a linear combination of C-blocks.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 3  # each block is 3x3
    
    # Strassen definitions:
    # (r, a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (block_row, block_col) to coefficient for A
    # b_coeffs: dict from (block_row, block_col) to coefficient for B
    # c_coeffs: dict from (block_row, block_col) to coefficient for C
    #
    # Block indices: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),  # C00: +1, C11: +1
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(0,0): -1, (1,1): -1}),  # C00: -1, C11: -1
        # M3 = A00 * (B01 - B11)
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): -1}),  # C01: +1, C11: -1
        # M4 = A11 * (B10 - B00)
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),  # C00: +1, C10: +1
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,1): 1, (1,1): 1}),  # C01: +1, C11: +1
        # M6 = (A10 - A00) * (B00 + B01)
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,0): 1}),  # C10: +1
        # M7 = (A01 - A11) * (B10 + B11)
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),  # C00: +1
    ]
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for r in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_steps[r]
        
        # For this Strassen step, we need to expand the block-level
        # linear combinations into element-level schoolbook terms.
        # 
        # A linear combination sum_{(br, bc)} a_coef[br,bc] * A_{br,bc}
        # where A_{br,bc} is a 3x3 block. The element at local position (lr, lc)
        # of block (br, bc) is at global position (br*3 + lr, bc*3 + lc).
        #
        # Similarly for B: sum_{(br, bc)} b_coef[br,bc] * B_{br,bc}
        # Element at local (lr, lc) of block (br, bc) is at global (br*3 + lr, bc*3 + lc).
        #
        # The product of these two linear combinations, expanded via schoolbook,
        # gives for each pair of local indices (lr_a, lc_a) for A and (lr_b, lc_b) for B:
        # The A-side element is at global row (br_a*3 + lr_a, bc_a*3 + lc_a) for each (br_a, bc_a) with coefficient a_coef
        # The B-side element is at global row (br_b*3 + lr_b, bc_b*3 + lc_b) for each (br_b, bc_b) with coefficient b_coef
        # The C-side contribution goes to block (br_c, bc_c) at local position (lr_a + lr_b - 1, lc_a + lc_b - 1)
        # wait, that's not right. Let me think more carefully.
        
        # Actually, the tensor decomposition works as follows:
        # The matrix multiplication tensor T[i,j,k] = delta_{j,k} (for C[i,k] = sum_j A[i,j]*B[j,k])
        # 
        # In the block-Strassen decomposition, each Strassen step r defines:
        # A_r = sum_{(br, bc)} a_r[br,bc] * A_{br,bc}  (a 3x3 "virtual" matrix, but really a linear combo)
        # B_r = sum_{(br, bc)} b_r[br,bc] * B_{br,bc}  (a 3x3 "virtual" matrix)
        # C_r = sum_{(br, bc)} c_r[br,bc] * C_{br,bc}  (a 3x3 "virtual" matrix)
        # And we need A_r * B_r = C_r (as 3x3 matrices)
        #
        # To decompose A_r * B_r = C_r into schoolbook rank-27 terms:
        # For each local row lr (0..2) and local col lc (0..2) of the 3x3 blocks:
        #   A_r[local row lr, local col lc] = sum over all (br, bc) of a_r[br,bc] * A[br*3+lr, bc*3+lc]
        #   B_r[local row lc, local col lc2] = sum over all (br, bc) of b_r[br,bc] * B[br*3+lc, bc*3+lc2]
        #   C_r[local row lr, local col lc2] = sum over all (br, bc) of c_r[br,bc] * C[br*3+lr, bc*3+lc2]
        #
        # The schoolbook decomposition of the 3x3 product X*Y=Z (where X, Y, Z are 3x3) is:
        # For each (lr, lc, lc2) in {0,1,2}^3:
        #   U[i*3 + lc, col] = X[local lr, local lc] coefficient
        #   V[lc*3 + lc2, col] = Y[local lc, local lc2] coefficient
        #   W[lc2*3 + lr, col] = Z[local lr, local lc2] coefficient
        #
        # But here X, Y, Z are linear combinations of blocks. So:
        # X[local lr, local lc] = sum_{(br, bc)} a_r[br,bc] * A_global[br*3+lr, bc*3+lc]
        # This means for a given (lr, lc, lc2) schoolbook term, the U factor is:
        #   For each (br, bc) with a_r[br,bc] != 0:
        #       U[(br*3+lr)*3 + (bc*3+lc), col] += a_r[br,bc]
        # Wait, I need to be more careful about the indexing.
        
        # Let me use different notation. Let's say the 6x6 matrix A has rows 0..5 and cols 0..5.
        # Block (br, bc) occupies rows br*3..br*3+2 and cols bc*3..bc*3+2.
        # So element (i, j) of A belongs to block (i//3, j//3) at local position (i%3, j%3).
        
        # For the 3x3 virtual matrix X = A_r:
        # X[lr, lc] = sum over all blocks (br, bc) that contribute to X at local position (lr, lc)
        # But each block A_{br,bc} is a 3x3 matrix. When we say A_r = sum a_r[br,bc] * A_{br,bc},
        # we're adding 3x3 matrices. So X[lr, lc] = sum_{(br,bc)} a_r[br,bc] * A_{br,bc}[lr, lc]
        # And A_{br,bc}[lr, lc] is the element at global position (br*3+lr, bc*3+lc).
        
        # So for the schoolbook term corresponding to local indices (lr, lc_a, lc_b):
        # (where lc_a is the col index in X/A-side, lc_b is the col index in Y/B-side,
        #  and lc_a must equal the row index in Y/B-side for the product to be defined)
        #
        # Actually, in standard matrix multiplication C = X * Y, C[lr, lc_b] = sum_lc X[lr, lc] * Y[lc, lc_b]
        # So the schoolbook term is indexed by (lr, lc, lc_b) where lc is the summation index.
        
        # For each (lr, lc, lc_b) in {0,1,2}^3:
        #   U-side: X[lr, lc] = sum_{(br, bc)} a_r[br,bc] * A_global[br*3+lr, bc*3+lc]
        #   So U[(br*3+lr)*m + (bc*3+lc), col] += a_r[br,bc] for each (br,bc)
        #   
        #   V-side: Y[lc, lc_b] = sum_{(br, bc)} b_r[br,bc] * B_global[br*3+lc, bc*3+lc_b]
        #   So V[(br*3+lc)*p + (bc*3+lc_b), col] += b_r[br,bc] for each (br,bc)
        #
        #   W-side: Z[lr, lc_b] = sum_{(br, bc)} c_r[br,bc] * C_global[br*3+lr, bc*3+lc_b]
        #   W index: k*n + i where k is the column index of C, i is the row index of C.
        #   C_global[br*3+lr, bc*3+lc_b] has row index = br*3+lr, col index = bc*3+lc_b
        #   So W[(bc*3+lc_b)*n + (br*3+lr), col] += c_r[br,bc] for each (br,bc)
        
        for lr in range(3):
            for lc in range(3):
                for lc_b in range(3):
                    # U side
                    for (br, bc), coeff_a in a_coeffs.items():
                        if coeff_a == 0:
                            continue
                        row_u = (br * 3 + lr) * 6 + (bc * 3 + lc)
                        U[row_u][col] += coeff_a
                    
                    # V side
                    for (br, bc), coeff_b in b_coeffs.items():
                        if coeff_b == 0:
                            continue
                        row_v = (br * 3 + lc) * 6 + (bc * 3 + lc_b)
                        V[row_v][col] += coeff_b
                    
                    # W side
                    for (br, bc), coeff_c in c_coeffs.items():
                        if coeff_c == 0:
                            continue
                        # C_global row = br*3+lr, col = bc*3+lc_b
                        # W index = k*n + i = col_C * n + row_C
                        row_w = (bc * 3 + lc_b) * 6 + (br * 3 + lr)
                        W[row_w][col] += coeff_c
                    
                    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Return the factor matrices for 6x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks of 3x3 matrices, achieving rank 189.
    """
    if time.time() > DEADLINE:
        # Fallback to schoolbook if we're over time (shouldn't happen for this simple construction)
        rank = 6 * 6 * 6
        U = [[0] * rank for _ in range(36)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        column = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i * 6 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 6 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return _build_strassen_block_decomposition()
