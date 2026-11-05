import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, p * n

    # Laderman's rank-15 algorithm for 3x3 matrix multiplication
    # We partition 6x6 into 3x3 grid of 2x2 blocks
    # Total rank = 15 (Laderman) * 4 (2x2 schoolbook) = 60

    # Laderman's 15 products for 3x3:
    # P1  = (A11 + A12 + A13) * (B11 + B12 + B13)
    # P2  = (A11 - A12 + A13) * (B11 - B12 + B13)
    # P3  = (A11 + A12 - A13) * (B11 + B12 - B13)
    # P4  = (A11 - A12 - A13) * (B11 - B12 - B13)
    # P5  = (A11 + 2*A12 + 3*A13) * (B11 + 2*B12 + 3*B13)
    # P6  = (A11 + 3*A12 + 2*A13) * (B11 + 3*B12 + 2*B13)
    # P7  = (A11 + 2*A12 - 3*A13) * (B11 + 2*B12 - 3*B13)
    # P8  = (A11 + 3*A12 - 2*A13) * (B11 + 3*B12 - 2*B13)
    # P9  = (A11 - 2*A12 + 3*A13) * (B11 - 2*B12 + 3*B13)
    # P10 = (A11 - 3*A12 + 2*A13) * (B11 - 3*B12 + 2*B13)
    # P11 = (A11 - 2*A12 - 3*A13) * (B11 - 2*B12 - 3*B13)
    # P12 = (A11 - 3*A12 - 2*A13) * (B11 - 3*B12 - 2*B13)
    # P13 = A13 * B31
    # P14 = A23 * B32
    # P15 = A33 * B33

    # For each product Pi, we need to compute:
    # Ci = sum over appropriate terms
    # The reconstruction formulas are:
    # C11 = (P1 + P2 + P3 + P4 + P5 + P6 - P7 - P8 - P9 - P10 + P11 + P12)/12
    # C12 = (P1 - P2 + P3 - P4 + 2*P5 - 3*P6 + 2*P7 - 3*P8 + P9 - P10 + P11 - P12)/12
    # C13 = (P1 + P2 - P3 - P4 + 3*P5 + 2*P6 - 3*P7 - 2*P8 - P9 - P10 - P11 - P12)/12
    # C21 = (P1 + P2 + P3 + P4 - 2*P5 - 3*P6 + 2*P7 + 3*P8 - P9 - P10 + P11 + P12)/12
    # C22 = (P1 - P2 - P3 + P4 - 3*P5 + 2*P6 - 3*P7 + 2*P8 + P9 - P10 + P11 - P12)/12
    # C23 = (P1 + P2 - P3 - P4 - 2*P5 + 3*P6 + 3*P7 - 2*P8 - P9 + P10 - P11 + P12)/12
    # C31 = (P1 + P2 + P3 + P4 - 3*P5 - 2*P6 + 3*P7 + 2*P8 + P9 + P10 - P11 - P12)/12
    # C32 = (P1 - P2 - P3 + P4 + 2*P5 - 3*P6 - 3*P7 + 2*P8 + P9 + P10 - P11 - P12)/12
    # C33 = (P1 + P2 - P3 - P4 - 3*P5 - 2*P6 + 3*P7 + 2*P8 + P9 + P10 - P11 - P12)/12

    # Wait, let me use the correct Laderman formulas. Let me look up the exact ones.
    # Actually, let me use a simpler approach: use the known rank-15 Laderman decomposition.
    # The 15 products and their contributions to the 9 output blocks are well-known.

    # Let me define the Laderman decomposition more carefully.
    # Each Pi is a product of two 2x2 blocks (linear combinations of the A and B blocks).
    # Then each output block Ci,j is a linear combination of the Pi's.

    # For 2x2 schoolbook, each 2x2 multiplication uses 4 scalar multiplications.
    # So total rank = 15 * 4 = 60.

    # Let me define the 15 Laderman products with their coefficients.
    # Using the standard Laderman (1976) rank-15 algorithm:

    # Product definitions (coefficients for A blocks and B blocks):
    # Each product Pi = (sum_a a_ia * A_i) * (sum_b b_jb * B_j)
    # where i,j are block indices (0-8 for 3x3 grid)

    # Laderman's 15 products:
    products = []

    # P1: (A0 + A1 + A2) * (B0 + B1 + B2)
    products.append(([1,1,1,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0]))
    # P2: (A0 - A1 + A2) * (B0 - B1 + B2)
    products.append(([1,-1,1,0,0,0,0,0,0], [1,-1,1,0,0,0,0,0,0]))
    # P3: (A0 + A1 - A2) * (B0 + B1 - B2)
    products.append(([1,1,-1,0,0,0,0,0,0], [1,1,-1,0,0,0,0,0,0]))
    # P4: (A0 - A1 - A2) * (B0 - B1 - B2)
    products.append(([1,-1,-1,0,0,0,0,0,0], [1,-1,-1,0,0,0,0,0,0]))
    # P5: (A0 + 2*A1 + 3*A2) * (B0 + 2*B1 + 3*B2)
    products.append(([1,2,3,0,0,0,0,0,0], [1,2,3,0,0,0,0,0,0]))
    # P6: (A0 + 3*A1 + 2*A2) * (B0 + 3*B1 + 2*B2)
    products.append(([1,3,2,0,0,0,0,0,0], [1,3,2,0,0,0,0,0,0]))
    # P7: (A0 + 2*A1 - 3*A2) * (B0 + 2*B1 - 3*B2)
    products.append(([1,2,-3,0,0,0,0,0,0], [1,2,-3,0,0,0,0,0,0]))
    # P8: (A0 + 3*A1 - 2*A2) * (B0 + 3*B1 - 2*B2)
    products.append(([1,3,-2,0,0,0,0,0,0], [1,3,-2,0,0,0,0,0,0]))
    # P9: (A0 - 2*A1 + 3*A2) * (B0 - 2*B1 + 3*B2)
    products.append(([1,-2,3,0,0,0,0,0,0], [1,-2,3,0,0,0,0,0,0]))
    # P10: (A0 - 3*A1 + 2*A2) * (B0 - 3*B1 + 2*B2)
    products.append(([1,-3,2,0,0,0,0,0,0], [1,-3,2,0,0,0,0,0,0]))
    # P11: (A0 - 2*A1 - 3*A2) * (B0 - 2*B1 - 3*B2)
    products.append(([1,-2,-3,0,0,0,0,0,0], [1,-2,-3,0,0,0,0,0,0]))
    # P12: (A0 - 3*A1 - 2*A2) * (B0 - 3*B1 - 2*B2)
    products.append(([1,-3,-2,0,0,0,0,0,0], [1,-3,-2,0,0,0,0,0,0]))
    # P13: A8 * B6  (A33 * B31)
    products.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0,0]))
    # P14: A5 * B7  (A23 * B32)
    products.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0]))
    # P15: A8 * B8  (A33 * B33)
    products.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))

    # Output reconstruction coefficients:
    # C[i,j] = sum_p coeff[i,j,p] * P_p
    # Let me use the correct Laderman output formulas.
    # This is getting complex. Let me use a different, verified approach.

    # Actually, let me just implement the 2x2 Strassen on 3x3 blocks properly,
    # but use a better base kernel. Or better yet, let me carefully implement
    # the Laderman decomposition with verified coefficients.

    # Let me use a simpler, verified rank-15 for 3x3. I'll use the known
    # decomposition from the literature.

    # Alternative: Use the fact that we can do 6x6 = 2x2 blocks of 3x3,
    # and use Strassen at the 2x2 level with 3x3 schoolbook (rank 27) inside.
    # That gives 7 * 27 = 189, which is what we have.

    # To get rank 60, we need Laderman (rank 15 for 3x3) with 2x2 schoolbook (rank 4) inside.
    # 15 * 4 = 60.

    # Let me carefully implement this. The key insight is:
    # - Partition 6x6 into 3x3 grid of 2x2 blocks
    # - Apply Laderman's rank-15 algorithm at the block level
    # - Each "block multiplication" in Laderman is actually a 2x2 matrix multiplication
    # - Use schoolbook (rank 4) for each 2x2 multiplication

    # The 15 Laderman products each involve linear combinations of 2x2 blocks.
    # For each product, we compute the 2x2 matrix product using 4 scalar multiplications.

    # Let me define the Laderman decomposition properly.
    # I'll use the standard formulation where:
    # - Block indices: (row, col) for 3x3 grid, so 9 blocks
    # - Block (r,c) corresponds to rows 2r:2r+2, cols 2c:2c+2 in the 6x6 matrix

    # Laderman's 15 products (verified from literature):
    # Let A_{ij} and B_{ij} be the 2x2 blocks.
    # Define:
    # P1 = (A11+A12+A13)(B11+B12+B13)
    # P2 = (A11-A12+A13)(B11-B12+B13)
    # P3 = (A11+A12-A13)(B11+B12-B13)
    # P4 = (A11-A12-A13)(B11-B12-B13)
    # P5 = (A11+2A12+3A13)(B11+2B12+3B13)
    # P6 = (A11+3A12+2A13)(B11+3B12+2B13)
    # P7 = (A11+2A12-3A13)(B11+2B12-3B13)
    # P8 = (A11+3A12-2A13)(B11+3B12-2B13)
    # P9 = (A11-2A12+3A13)(B11-2B12+3B13)
    # P10 = (A11-3A12+2A13)(B11-3B12+2B13)
    # P11 = (A11-2A12-3A13)(B11-2B12-3B13)
    # P12 = (A11-3A12-2A13)(B11-3B12-2B13)
    # P13 = A13*B31
    # P14 = A23*B32
    # P15 = A33*B33

    # Output:
    # C11 = (P1+P2+P3+P4+P5+P6-P7-P8-P9-P10+P11+P12)/12
    # C12 = (P1-P2+P3-P4+2P5-3P6+2P7-3P8+P9-P10+P11-P12)/12
    # C13 = (P1+P2-P3-P4+3P5+2P6-3P7-2P8-P9-P10-P11-P12)/12
    # C21 = (P1+P2+P3+P4-2P5-3P6+2P7+3P8-P9-P10+P11+P12)/12
    # C22 = (P1-P2-P3+P4-3P5+2P6-3P7+2P8+P9-P10+P11-P12)/12
    # C23 = (P1+P2-P3-P4-2P5+3P6+3P7-2P8-P9+P10-P11+P12)/12
    # C31 = (P1+P2+P3+P4-3P5-2P6+3P7+2P8+P9+P10-P11-P12)/12
    # C32 = (P1-P2-P3+P4+2P5-3P6-3P7+2P8+P9+P10-P11-P12)/12
    # C33 = (P1+P2-P3-P4-3P5-2P6+3P7+2P8+P9+P10-P11-P12)/12

    # Wait, I need to double-check these. Let me use a verified source.
    # Actually, the Laderman decomposition I'm trying to recall might not be exactly right.
    # Let me use a different, simpler approach that I can verify.

    # Let me just use the 2x2 Strassen with 3x3 blocks (rank 189) but verify it works,
    # and also try to implement a correct Laderman-based decomposition.

    # For now, let me implement the Laderman decomposition carefully and verify it.
    # If verification fails, fall back to Strassen (189) or schoolbook (216).

    total_rank = 60  # 15 Laderman products * 4 (2x2 schoolbook)
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    # 2x2 schoolbook: for multiplying 2x2 matrices
    # U2[i*2+j, col] = 1 for the i,j entry
    # V2[j*2+k, col] = 1 for the j,k entry
    # W2[k*2+i, col] = 1 for the k,i entry (note: W index is k*n+i where n=2 here)
    # Actually for 2x2: W index is k*2 + i

    def add_2x2_product(base_col, a_coeff, b_coeff, c_coeff, block_a, block_b, block_c):
        """Add a 2x2 schoolbook product term.
        block_a, block_b, block_c are (row_block, col_block) indices in the 3x3 grid.
        a_coeff, b_coeff, c_coeff are scalar coefficients.
        """
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = base_col + i * 4 + j * 2 + k
                    # U: A-side, global index (block_a_row*2 + i, block_a_col*2 + j)
                    gi_a = block_a[0] * 2 + i
                    gj_a = block_a[1] * 2 + j
                    U6[gi_a * 6 + gj_a, col] += a_coeff

                    # V: B-side, global index (block_b_row*2 + j, block_b_col*2 + k)
                    gj_b = block_b[0] * 2 + j
                    gk_b = block_b[1] * 2 + k
                    V6[gj_b * 6 + gk_b, col] += b_coeff

                    # W: C-side, global index (k*6 + i) -> actually W[k*6 + i] where k is from B, i is from A
                    # Wait, the convention is W[k*n+i, r] where n=6
                    # For the 2x2 sub-block at position (block_c_row, block_c_col):
                    # The output block C[block_c_row, block_c_col] has entries C[2*br+ki, 2*bc+ii]
                    # where ki is the row index within the 2x2 block (from B's column),
                    # and ii is the column index within the 2x2 block (from A's row)
                    # W index: k_global * 6 + i_global
                    # k_global = block_c_col * 2 + k  (k is the column index in B, which becomes row in C)
                    # i_global = block_c_row * 2 + i  (i is the row index in A, which becomes col in C)
                    gk_c = block_c[1] * 2 + k
                    gi_c = block_c[0] * 2 + i
                    W6[gk_c * 6 + gi_c, col] += c_coeff

    # Now apply Laderman's 15 products
    # Block indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
    # But we need (row, col) tuples for the blocks.

    # Laderman products:
    # P1: A blocks with coeff [1,1,1,0,0,0,0,0,0] for A11,A12,A13
    #     B blocks with coeff [1,1,1,0,0,0,0,0,0] for B11,B12,B13
    #     This is (A11+A12+A13) * (B11+B12+B13)
    #     But this is a sum of 9 2x2 products! That's not how it works.
    #     
    # Wait, I'm confusing myself. In Laderman's algorithm, each Pi is a single
    # matrix product where the matrices are linear combinations of blocks.
    # So P1 = (A11+A12+A13) * (B11+B12+B13) is ONE matrix product of two 3x3 matrices,
    # but when we decompose into 2x2 blocks, this becomes a sum of 9 2x2 products.
    #
    # That's not right either. Let me think again.
    #
    # In Laderman's algorithm for 3x3 matrix multiplication:
    # - A and B are 3x3 matrices
    # - The 15 products Pi are each products of two 3x3 matrices (linear combinations)
    # - Each Pi is a 3x3 matrix
    # - The output C is a linear combination of the Pi's
    #
    # But in our case, we're partitioning 6x6 into 3x3 grid of 2x2 blocks.
    # So the "3x3 matrix" in Laderman's algorithm is actually a 3x3 grid of 2x2 blocks.
    # Each "entry" in the 3x3 Laderman matrix is a 2x2 block.
    #
    # So when Laderman says P1 = (A11+A12+A13)(B11+B12+B13), this means:
    # P1 is a 3x3 matrix where each entry is a 2x2 block.
    # P1[i,j] = sum over all (a,b) such that... no wait.
    #
    # Actually, in Laderman's algorithm, A11, A12, etc. are individual entries (scalars)
    # of the 3x3 matrix. So P1 = (A11+A12+A13)(B11+B12+B13) is a single scalar product.
    #
    # But in our block decomposition, A11 is a 2x2 block, not a scalar.
    # So we need to interpret Laderman's algorithm at the block level.
    #
    # The correct interpretation: Laderman's algorithm works on 3x3 matrices where
    # each "entry" is a scalar. If we replace each scalar with a 2x2 block, then
    # each scalar product becomes a 2x2 matrix product (which requires 4 scalar multiplications
    # in schoolbook).
    #
    # So P1 = (A11+A12+A13)(B11+B12+B13) where A11, A12, A13 are 2x2 blocks means:
    # P1 = (A11+A12+A13) * (B11+B12+B13) is a 2x2 matrix product.
    # This requires 4 scalar multiplications.
    #
    # So total rank = 15 * 4 = 60.

    # Now, the 15 Laderman products at the block level:
    # Each Pi = (sum of some A blocks) * (sum of some B blocks), where each block is 2x2.
    # The result Pi is a 2x2 matrix.
    # The output blocks Cij are linear combinations of the Pi's.

    # Let me define the 15 products with their block coefficients:
    # Block indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)

    laderman_products = [
        # (A_coeffs, B_coeffs) where coeffs are dicts mapping (row,col) -> coeff
        # P1
        ({(0,0):1, (0,1):1, (0,2):1}, {(0,0):1, (0,1):1, (0,2):1}),
        # P2
        ({(0,0):1, (0,1):-1, (0,2):1}, {(0,0):1, (0,1):-1, (0,2):1}),
        # P3
        ({(0,0):1, (0,1):1, (0,2):-1}, {(0,0):1, (0,1):1, (0,2):-1}),
        # P4
        ({(0,0):1, (0,1):-1, (0,2):-1}, {(0,0):1, (0,1):-1, (0,2):-1}),
        # P5
        ({(0,0):1, (0,1):2, (0,2):3}, {(0,0):1, (0,1):2, (0,2):3}),
        # P6
        ({(0,0):1, (0,1):3, (0,2):2}, {(0,0):1, (0,1):3, (0,2):2}),
        # P7
        ({(0,0):1, (0,1):2, (0,2):-3}, {(0,0):1, (0,1):2, (0,2):-3}),
        # P8
        ({(0,0):1, (0,1):3, (0,2):-2}, {(0,0):1, (0,1):3, (0,2):-2}),
        # P9
        ({(0,0):1, (0,1):-2, (0,2):3}, {(0,0):1, (0,1):-2, (0,2):3}),
        # P10
        ({(0,0):1, (0,1):-3, (0,2):2}, {(0,0):1, (0,1):-3, (0,2):2}),
        # P11
        ({(0,0):1, (0,1):-2, (0,2):-3}, {(0,0):1, (0,1):-2, (0,2):-3}),
        # P12
        ({(0,0):1, (0,1):-3, (0,2):-2}, {(0,0):1, (0,1):-3, (0,2):-2}),
        # P13
        ({(0,2):1}, {(2,0):1}),
        # P14
        ({(1,2):1}, {(2,1):1}),
        # P15
        ({(2,2):1}, {(2,2):1}),
    ]

    # Output reconstruction: C[i,j] = sum_p coeff[i,j,p] * P_p
    # The coefficients are 1/12 times the following:
    output_coeffs = {
        # (out_row, out_col): {prod_idx: coeff}
        (0,0): {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:-1, 7:-1, 8:-1, 9:-1, 10:1, 11:1},
        (0,1): {0:1, 1:-1, 2:1, 3:-1, 4:2, 5:-3, 6:2, 7:-3, 8:1, 9:-1, 10:1, 11:-1},
        (0,2): {0:1, 1:1, 2:-1, 3:-1, 4:3, 5:2, 6:-3, 7:-2, 8:-1, 9:-1, 10:-1, 11:-1},
        (1,0): {0:1, 1:1, 2:1, 3:1, 4:-2, 5:-3, 6:2, 7:3, 8:-1, 9:-1, 10:1, 11:1},
        (1,1): {0:1, 1:-1, 2:-1, 3:1, 4:-3, 5:2, 6:-3, 7:2, 8:1, 9:-1, 10:1, 11:-1},
        (1,2): {0:1, 1:1, 2:-1, 3:-1, 4:-2, 5:3, 6:3, 7:-2, 8:-1, 9:1, 10:-1, 11:1},
        (2,0): {0:1, 1:1, 2:1, 3:1, 4:-3, 5:-2, 6:3, 7:2, 8:1, 9:1, 10:-1, 11:-1},
        (2,1): {0:1, 1:-1, 2:-1, 3:1, 4:2, 5:-3, 6:-3, 7:2, 8:1, 9:1, 10:-1, 11:-1},
        (2,2): {0:1, 1:1, 2:-1, 3:-1, 4:-3, 5:-2, 6:3, 7:2, 8:1, 9:1, 10:-1, 11:-1},
    }

    # Scale factor: 1/12 for products 1-12, 1 for products 13-15
    scale = [1/12]*12 + [1]*3

    # Now build the factor matrices
    for p_idx, (a_coeffs, b_coeffs) in enumerate(laderman_products):
        base_col = p_idx * 4  # 4 scalar multiplications per 2x2 product
        a_coeff_val = scale[p_idx]
        b_coeff_val = scale[p_idx]

        # For each pair of A block and B block in this product:
        for (ar, ac), a_c in a_coeffs.items():
            for (br, bc), b_c in b_coeffs.items():
                # This contributes to each output block C[i,j] with coefficient:
                # sum over p of coeff[i,j,p] * a_c * b_c * scale[p]
                # But we need to add the 2x2 schoolbook for this specific (A_block, B_block) pair
                # to each output block that this product contributes to.

                # For each output block (oi, oj) that product p_idx contributes to:
                for (oi, oj), coeff_dict in output_coeffs.items():
                    if p_idx in coeff_dict:
                        c_coeff = coeff_dict[p_idx] * a_c * b_c * scale[p_idx]
                        if c_coeff != 0:
                            add_2x2_product(base_col, a_coeff_val * a_c, b_coeff_val * b_c, c_coeff,
                                           (ar, ac), (br, bc), (oi, oj))

    # Wait, this is getting complicated and I'm not sure the coefficients are right.
    # Let me just verify numerically and fall back if it doesn't work.

    # Verify
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            if np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    # Fallback: 2x2 Strassen with 3x3 blocks (rank 189)
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)
            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val
            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val
            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    add_strassen_term(0, [(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])
    add_strassen_term(r3, [(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])
    add_strassen_term(2*r3, [(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)])
    add_strassen_term(3*r3, [(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)])
    add_strassen_term(4*r3, [(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)])
    add_strassen_term(5*r3, [(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    add_strassen_term(6*r3, [(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])

    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            if np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    # Final fallback: schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
