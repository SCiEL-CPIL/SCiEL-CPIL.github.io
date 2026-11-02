"""Matrix multiplication 4,4,9 using Strassen's algorithm for 4x4 blocks."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = 0
    U = []
    V = []
    W = []

    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        U.append([0] * (rank + 1))
        V.append([0] * (rank + 1))
        W.append([0] * (rank + 1))
        U[u_row][rank] = u_val
        V[v_row][rank] = v_val
        W[w_row][rank] = w_val
        rank += 1

    # Helper: add schoolbook terms for 2x2x2 multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # We need to express the multiplication of linear combinations
    # For a term in the expansion, we have:
    # C[i, j] += A[i, k] * B[k, j]
    # But A and B are linear combinations of blocks of the original matrices.
    # We'll pass the indices of the blocks directly.

    # Strassen's algorithm for 4x4 = (2x2 blocks of 2x2 matrices)
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = A * B
    # 
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each of the 7 Strassen terms, we decompose the 2x2x2 multiplication using schoolbook.
    # The "matrices" in each Strassen term are linear combinations of 2x2 blocks.
    # We need to expand each term into rank-1 terms of the original multiplication.

    # Let's define the block indices:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)

    # For a 2x2x2 multiplication of linear combinations:
    # If X = sum_i alpha_i * A_i (where A_i is a 2x2 block)
    # and Y = sum_j beta_j * B_j (where B_j is a 2x2 block)
    # Then Z = X * Y = sum_{i,j} alpha_i * beta_j * (A_i * B_j)
    # Each (A_i * B_j) is a 2x2x2 multiplication, which has 8 rank-1 terms.

    # Let's define a helper to add the schoolbook decomposition of a product of two linear combinations
    # of 2x2 blocks.

    def add_strassen_term(a_coeffs, b_coeffs, c_row_offset, c_col_offset):
        """
        a_coeffs: list of (block_index, coeff) for A-side linear combination
        b_coeffs: list of (block_index, coeff) for B-side linear combination
        c_row_offset, c_col_offset: not used directly, we handle the C placement separately
        """
        # a_coeffs: list of (block, coeff)
        # b_coeffs: list of (block, coeff)
        # For each pair (a_block, a_coeff) and (b_block, b_coeff):
        #   Add schoolbook terms for A[a_block] * B[b_block]
        #   The result is added to the appropriate C block with coefficient a_coeff * b_coeff

        for a_block, a_coeff in a_coeffs:
            for b_block, b_coeff in b_coeffs:
                total_coeff = a_coeff * b_coeff
                if total_coeff == 0:
                    continue
                # A[a_block] * B[b_block] is a 2x2x2 multiplication
                # A[a_block] has rows (ar, ar+1) and cols (ac, ac+1)
                # B[b_block] has rows (br, br+1) and cols (bc, bc+1)
                # The product contributes to C at the appropriate position

                # Determine the global row/col ranges
                # A blocks: 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
                # B blocks: 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)

                a_row_start = (a_block // 2) * 2
                a_col_start = (a_block % 2) * 2
                b_row_start = (b_block // 2) * 2
                b_col_start = (b_block % 2) * 2

                # For each (i, k, l) in 0..1:
                # U row: global A index = (a_row_start + i) * m + (a_col_start + k)
                # V row: global B index = (b_row_start + k) * p + (b_col_start + l)
                # W row: global C index = (c_row_start + l) * n + (c_col_start + i)
                # But we need to figure out where in C this product lands.

                # Actually, the product A[a_block] * B[b_block] is a 2x2 matrix.
                # It contributes to C at the position determined by the Strassen formula.
                # We'll handle the C placement in the outer loop.

                # For now, let's just store the terms and apply them later.
                # Actually, let's just add the rank-1 terms directly.

                for i in range(2):
                    for k in range(2):
                        for l in range(2):
                            u_row = (a_row_start + i) * m + (a_col_start + k)
                            v_row = (b_row_start + k) * p + (b_col_start + l)
                            # c_row and c_col depend on which C block this contributes to
                            # We'll pass that in from the caller
                            yield (u_row, v_row, i, k, l, total_coeff, a_block, b_block)

    # Let's restructure: we'll compute the Strassen terms and for each, expand into rank-1 terms.

    # Define the Strassen terms:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)

    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each C block, we sum contributions from the relevant M terms.

    # Let's define the contributions to each C block:
    # C11: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12: M3(+1), M5(+1)
    # C21: M2(+1), M4(+1)
    # C22: M1(+1), M2(-1), M3(+1), M6(+1)

    # Define M terms as (a_coeffs, b_coeffs)
    # Block indices: 0=A11/B11, 1=A12/B12, 2=A21/B21, 3=A22/B22

    M1_a = [(0, 1), (3, 1)]
    M1_b = [(0, 1), (3, 1)]

    M2_a = [(2, 1), (3, 1)]
    M2_b = [(0, 1)]

    M3_a = [(0, 1)]
    M3_b = [(1, 1), (3, -1)]

    M4_a = [(3, 1)]
    M4_b = [(2, 1), (0, -1)]

    M5_a = [(0, 1), (1, 1)]
    M5_b = [(3, 1)]

    M6_a = [(2, 1), (0, -1)]
    M6_b = [(0, 1), (1, 1)]

    M7_a = [(1, 1), (3, -1)]
    M7_b = [(2, 1), (3, 1)]

    # Now, for each C block, sum the contributions:
    # C11: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12: M3(+1), M5(+1)
    # C21: M2(+1), M4(+1)
    # C22: M1(+1), M2(-1), M3(+1), M6(+1)

    # C block indices: 0=C11 (rows 0-1, cols 0-1), 1=C12 (rows 0-1, cols 2-3), 2=C21 (rows 2-3, cols 0-1), 3=C22 (rows 2-3, cols 2-3)

    c_blocks = {
        0: [(1, M1_a, M1_b), (1, M4_a, M4_b), (-1, M5_a, M5_b), (1, M7_a, M7_b)],
        1: [(1, M3_a, M3_b), (1, M5_a, M5_b)],
        2: [(1, M2_a, M2_b), (1, M4_a, M4_b)],
        3: [(1, M1_a, M1_b), (-1, M2_a, M2_b), (1, M3_a, M3_b), (1, M6_a, M6_b)],
    }

    for c_block_idx, contributions in c_blocks.items():
        c_row_start = (c_block_idx // 2) * 2
        c_col_start = (c_block_idx % 2) * 2
        for sign, a_coeffs, b_coeffs in contributions:
            for a_block, a_coeff in a_coeffs:
                for b_block, b_coeff in b_coeffs:
                    total_coeff = sign * a_coeff * b_coeff
                    if total_coeff == 0:
                        continue
                    a_row_start = (a_block // 2) * 2
                    a_col_start = (a_block % 2) * 2
                    b_row_start = (b_block // 2) * 2
                    b_col_start = (b_block % 2) * 2
                    for i in range(2):
                        for k in range(2):
                            for l in range(2):
                                u_row = (a_row_start + i) * m + (a_col_start + k)
                                v_row = (b_row_start + k) * p + (b_col_start + l)
                                w_row = (c_row_start + l) * n + (c_col_start + i)
                                add_term(u_row, v_row, w_row, total_coeff, 1, 1)

    # Now handle the last column of B (col 3, which is the 9th column, index 8)
    # B's 9th column is a 4x1 vector.
    # C's 4th column (col 3) = A * B[:, 3]
    # This is a 4x4 times 4x1 multiplication.
    # We can decompose this using schoolbook: 16 rank-1 terms.
    for i in range(4):
        for k in range(4):
            u_row = i * m + k
            v_row = k * p + 8  # 9th column, index 8
            w_row = 3 * n + i  # 4th column of C, index 3
            add_term(u_row, v_row, w_row, 1, 1, 1)

    # Convert to lists of lists
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
