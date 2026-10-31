"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Rank-49 recursive Strassen decomposition for 4x4 x 4x4 matrix multiplication.

    Output convention: W[k*n + i, c].
    U[i*m + j, c], V[j*p + k, c].
    """
    n, m, p = 4, 4, 4

    # We build the decomposition recursively.
    # Level 1: 4x4 = 7 blocks of 2x2 (Strassen).
    # Level 2: each 2x2 = 7 scalar products (Strassen base case).
    # Total rank = 7 * 7 = 49.

    # Strassen base case for 2x2:
    # Given A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Strassen for 4x4 (block level):
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # N1 = (A11+A22)(B11+B22)
    # N2 = (A21+A22)B11
    # N3 = A11(B12-B22)
    # N4 = A22(B21-B11)
    # N5 = (A11+A12)B22
    # N6 = (A21-A11)(B11+B12)
    # N7 = (A12-A22)(B21+B22)
    # C11 = N1 + N4 - N5 + N7
    # C12 = N3 + N5
    # C21 = N2 + N4
    # C22 = N1 - N2 + N3 + N6

    # Each Nj is a 2x2 product, decomposed into 7 scalar products.
    # So total rank = 7 * 7 = 49.

    # We'll build U, V, W as dictionaries mapping (row, col) -> value.
    # col index = block_col * 7 + base_col, where block_col in 0..6, base_col in 0..6.

    U = {}
    V = {}
    W = {}

    def add_U(i, j, col, val):
        key = (i * m + j, col)
        U[key] = U.get(key, 0) + val

    def add_V(j, k, col, val):
        key = (j * p + k, col)
        V[key] = V.get(key, 0) + val

    def add_W(k, i, col, val):
        key = (k * n + i, col)
        W[key] = W.get(key, 0) + val

    # For each of the 7 block-level Strassen products Nj, we need to express
    # the A-side and B-side as linear combinations of 2x2 blocks, then each
    # block as a linear combination of scalars.

    # Block-level Strassen definitions (each Nj = (A_side)(B_side) where
    # A_side and B_side are 2x2 matrices expressed as linear combos of A-blocks):

    # N1: A_side = A11 + A22, B_side = B11 + B22
    # N2: A_side = A21 + A22, B_side = B11
    # N3: A_side = A11,        B_side = B12 - B22
    # N4: A_side = A22,        B_side = B21 - B11
    # N5: A_side = A11 + A12,  B_side = B22
    # N6: A_side = A21 - A11,  B_side = B11 + B12
    # N7: A_side = A12 - A22,  B_side = B21 + B22

    # For each Nj, the 2x2 product is decomposed via base-case Strassen.
    # Let A_side = sum of terms (coeff * A_block), B_side = sum of terms (coeff * B_block).
    # The base Strassen computes M1..M7 from A_side and B_side entries.
    # Each Mk is a product of a linear combo of A_side entries and a linear combo of B_side entries.

    # Let's define the A_side and B_side as 2x2 matrices with entries that are
    # linear combinations of the original A-block entries (a_{i,j}) and B-block entries (b_{i,j}).

    # A-blocks: A11, A12, A21, A22 (each is 2x2)
    # B-blocks: B11, B12, B21, B22 (each is 2x2)

    # For block Nj, A_side is a 2x2 matrix where each entry is a linear combo of A-block entries.
    # Let's represent A_side[r][c] = sum of (coeff * A_block[r'][c'][i][j]) for various.

    # Actually, let me think more carefully. A11 is the top-left 2x2 block of A.
    # A11[i][j] = A[i][j] for i in {0,1}, j in {0,1}.
    # A12[i][j] = A[i][j+2] for i in {0,1}, j in {0,1}.
    # A21[i][j] = A[i+2][j] for i in {0,1}, j in {0,1}.
    # A22[i][j] = A[i+2][j+2] for i in {0,1}, j in {0,1}.

    # Similarly for B.

    # So A_side is a 2x2 matrix whose entries are linear combinations of A's entries.
    # A_side[r][c] = sum over blocks B_blk in {11,12,21,22} of coeff[B_blk] * A_blk[r][c]
    # where A_blk[r][c] maps to A[i][j] with i = (blk_row*2 + r), j = (blk_col*2 + c).

    # Let me define for each Nj:
    # A_side_coeffs: dict from (blk_row, blk_col) -> coeff, for the A-side 2x2 matrix
    # B_side_coeffs: dict from (blk_row, blk_col) -> coeff, for the B-side 2x2 matrix

    # Then A_side[r][c] = sum_{(br,bc)} A_side_coeffs[(br,bc)] * A[br*2+r][bc*2+c]
    # B_side[r][c] = sum_{(br,bc)} B_side_coeffs[(br,bc)] * B[br*2+r][bc*2+c]

    # The base Strassen then computes M1..M7 from A_side and B_side.
    # Each Mk is a product of two linear combos.
    # The result C (2x2) = combination of M1..M7.
    # C[r][c] = sum_{mk=1}^{7} C_coeffs[mk][r][c] * Mk

    # And Mk = (linear combo of A_side entries) * (linear combo of B_side entries)
    # A_side entry (ar, ac) = sum_{(br,bc)} A_side_coeffs[(br,bc)] * A[br*2+ar][bc*2+ac]
    # So Mk = sum_{(ar,ac),(br,bc)} A_side_coeffs[(br,bc)] * A_side_entry_coeff[ar][ac] * A[br*2+ar][bc*2+ac]
    #         * sum_{(ar2,ac2),(br2,bc2)} B_side_coeffs[(br2,bc2)] * B_side_entry_coeff[ar2][ac2] * B[br2*2+ar2][bc2*2+ac2]

    # This is getting complex. Let me just build it directly.

    # Base Strassen M1..M7 definitions (A_side = [[a00,a01],[a10,a11]], B_side = [[b00,b01],[b10,b11]]):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Mk, define the A-side linear combo and B-side linear combo:
    # M1: A = a00 + a11, B = b00 + b11
    # M2: A = a10 + a11, B = b00
    # M3: A = a00,        B = b01 - b11
    # M4: A = a11,        B = b10 - b00
    # M5: A = a00 + a01, B = b11
    # M6: A = a10 - a00, B = b00 + b01
    # M7: A = a01 - a11, B = b10 + b11

    # C_coeffs (2x2 for each Mk):
    # M1: [[1, 0], [0, 1]]
    # M2: [[0, 0], [1, -1]]
    # M3: [[0, 1], [0, 1]]
    # M4: [[1, 0], [1, 0]]
    # M5: [[-1, 1], [0, 0]]
    # M6: [[0, 0], [0, 1]]
    # M7: [[1, 0], [0, 0]]

    # Block-level C_coeffs (which Nj goes into which output block position):
    # C11 = N1 + N4 - N5 + N7
    # C12 = N3 + N5
    # C21 = N2 + N4
    # C22 = N1 - N2 + N3 + N6
    # So output block (bi, bj) gets contributions from Nj with coefficient:
    # (0,0): N1:+1, N4:+1, N5:-1, N7:+1
    # (0,1): N3:+1, N5:+1
    # (1,0): N2:+1, N4:+1
    # (1,1): N1:+1, N2:-1, N3:+1, N6:+1

    # Now, for each Nj, the A_side and B_side are 2x2 matrices defined by:
    # A_side[r][c] = sum_{(br,bc)} A_side_coeffs_Nj[(br,bc)] * A[br*2+r][bc*2+c]
    # B_side[r][c] = sum_{(br,bc)} B_side_coeffs_Nj[(br,bc)] * B[br*2+r][bc*2+c]

    # A_side_coeffs for each Nj (mapping (br,bc) -> coeff):
    # N1: A_side = A11 + A22, so A_side[r][c] = A[0*2+r][0*2+c] + A[1*2+r][1*2+c]
    #     coeffs: {(0,0):1, (1,1):1}
    # N2: A_side = A21 + A22, so A_side[r][c] = A[1*2+r][0*2+c] + A[1*2+r][1*2+c]
    #     coeffs: {(1,0):1, (1,1):1}
    # N3: A_side = A11, so A_side[r][c] = A[0*2+r][0*2+c]
    #     coeffs: {(0,0):1}
    # N4: A_side = A22, so A_side[r][c] = A[1*2+r][1*2+c]
    #     coeffs: {(1,1):1}
    # N5: A_side = A11 + A12, so A_side[r][c] = A[0*2+r][0*2+c] + A[0*2+r][1*2+c]
    #     coeffs: {(0,0):1, (0,1):1}
    # N6: A_side = A21 - A11, so A_side[r][c] = A[1*2+r][0*2+c] - A[0*2+r][0*2+c]
    #     coeffs: {(1,0):1, (0,0):-1}
    # N7: A_side = A12 - A22, so A_side[r][c] = A[0*2+r][1*2+c] - A[1*2+r][1*2+c]
    #     coeffs: {(0,1):1, (1,1):-1}

    # B_side_coeffs for each Nj:
    # N1: B_side = B11 + B22, coeffs: {(0,0):1, (1,1):1}
    # N2: B_side = B11, coeffs: {(0,0):1}
    # N3: B_side = B12 - B22, coeffs: {(0,1):1, (1,1):-1}
    # N4: B_side = B21 - B11, coeffs: {(1,0):1, (0,0):-1}
    # N5: B_side = B22, coeffs: {(1,1):1}
    # N6: B_side = B11 + B12, coeffs: {(0,0):1, (0,1):1}
    # N7: B_side = B21 + B22, coeffs: {(1,0):1, (1,1):1}

    A_side_coeffs = {
        1: {(0,0):1, (1,1):1},
        2: {(1,0):1, (1,1):1},
        3: {(0,0):1},
        4: {(1,1):1},
        5: {(0,0):1, (0,1):1},
        6: {(1,0):1, (0,0):-1},
        7: {(0,1):1, (1,1):-1},
    }
    B_side_coeffs = {
        1: {(0,0):1, (1,1):1},
        2: {(0,0):1},
        3: {(0,1):1, (1,1):-1},
        4: {(1,0):1, (0,0):-1},
        5: {(1,1):1},
        6: {(0,0):1, (0,1):1},
        7: {(1,0):1, (1,1):1},
    }

    # Base Strassen Mk definitions:
    # Each Mk = (sum of A_side entries with coeffs) * (sum of B_side entries with coeffs)
    # A_side entry (ar, ac) has coeff from A_side_coeffs
    # So Mk = sum_{ar,ac,br,bc} a_coeff[ar,ac] * A_side_coeffs[(br,bc)] * A[br*2+ar][bc*2+ac]
    #         * sum_{ar2,ac2,br2,bc2} b_coeff[ar2,ac2] * B_side_coeffs[(br2,bc2)] * B[br2*2+ar2][bc2*2+ac2]

    # Mk A-side entry coeffs (ar, ac) -> coeff:
    M_A = {
        1: {(0,0):1, (1,1):1},
        2: {(1,0):1, (1,1):1},
        3: {(0,0):1},
        4: {(1,1):1},
        5: {(0,0):1, (0,1):1},
        6: {(1,0):1, (0,0):-1},
        7: {(0,1):1, (1,1):-1},
    }
    # Mk B-side entry coeffs (ar, ac) -> coeff:
    M_B = {
        1: {(0,0):1, (1,1):1},
        2: {(0,0):1},
        3: {(0,1):1, (1,1):-1},
        4: {(1,0):1, (0,0):-1},
        5: {(1,1):1},
        6: {(0,0):1, (0,1):1},
        7: {(1,0):1, (1,1):1},
    }

    # C_coeffs for base Strassen: C[r][c] = sum_mk C_coeffs[mk][r][c] * Mk
    C_base = {
        1: {(0,0):1, (0,1):0, (1,0):0, (1,1):1},
        2: {(0,0):0, (0,1):0, (1,0):1, (1,1):-1},
        3: {(0,0):0, (0,1):1, (1,0):0, (1,1):1},
        4: {(0,0):1, (0,1):0, (1,0):1, (1,1):0},
        5: {(0,0):-1, (0,1):1, (1,0):0, (1,1):0},
        6: {(0,0):0, (0,1):0, (1,0):0, (1,1):1},
        7: {(0,0):1, (0,1):0, (1,0):0, (1,1):0},
    }

    # Block-level C_coeffs: output block (bi, bj) gets contribution from Nj with coeff:
    C_block = {
        (0,0): {1:1, 4:1, 5:-1, 7:1},
        (0,1): {3:1, 5:1},
        (1,0): {2:1, 4:1},
        (1,1): {1:1, 2:-1, 3:1, 6:1},
    }

    # Now build U, V, W.
    # For each Nj (j=1..7), for each Mk (k=1..7), for each (ar,ac) in M_A[k], (ar2,ac2) in M_B[k]:
    #   col = (j-1)*7 + (k-1)
    #   For A side: A[br*2+ar][bc*2+ac] contributes with coeff = A_side_coeffs[j][(br,bc)] * M_A[k][(ar,ac)]
    #   For B side: B[br2*2+ar2][bc2*2+ac2] contributes with coeff = B_side_coeffs[j][(br2,bc2)] * M_B[k][(ar2,ac2)]
    #   Then C_block[(bi,bj)][j] * C_base[k][(cr,cc)] gives the contribution to output.
    #   Output C[bi*2+cr][bj*2+cc] gets this.
    #   In W convention: W[k_idx*n + i_idx, col] where i_idx = bi*2+cr, k_idx = bj*2+cc.
    #   Wait, W is indexed by [k*n + i] where k is the column index of B (and output), i is the row index of A (and output).
    #   Output C[i][j] where i is row (from A), j is column (from B).
    #   W[k*n + i, col] where k = j (column), i = i (row).
    #   So W[j*n + i, col].

    for j in range(1, 8):
        for k in range(1, 8):
            col = (j - 1) * 7 + (k - 1)

            # A side: for each (ar, ac) in M_A[k], and each (br, bc) in A_side_coeffs[j]:
            #   A[br*2+ar][bc*2+ac] gets coeff = A_side_coeffs[j][(br,bc)] * M_A[k][(ar,ac)]
            for (ar, ac), a_coeff in M_A[k].items():
                for (br, bc), b_coeff in A_side_coeffs[j].items():
                    i_idx = br * 2 + ar
                    j_idx = bc * 2 + ac
                    val = b_coeff * a_coeff
                    if val != 0:
                        add_U(i_idx, j_idx, col, val)

            # B side: for each (ar2, ac2) in M_B[k], and each (br2, bc2) in B_side_coeffs[j]:
            #   B[br2*2+ar2][bc2*2+ac2] gets coeff = B_side_coeffs[j][(br2,bc2)] * M_B[k][(ar2,ac2)]
            for (ar2, ac2), a_coeff in M_B[k].items():
                for (br2, bc2), b_coeff in B_side_coeffs[j].items():
                    i_idx = br2 * 2 + ar2
                    j_idx = bc2 * 2 + ac2
                    val = b_coeff * a_coeff
                    if val != 0:
                        add_V(i_idx, j_idx, col, val)

            # W side: for each (bi, bj) in C_block, if j is in C_block[(bi,bj)]:
            #   block_coeff = C_block[(bi,bj)][j]
            #   for each (cr, cc) in C_base[k]:
            #     base_coeff = C_base[k][(cr,cc)]
            #     i_idx = bi*2 + cr, k_idx = bj*2 + cc
            #     W[k_idx*n + i_idx, col] += block_coeff * base_coeff
            for (bi, bj), j_coeffs in C_block.items():
                if j not in j_coeffs:
                    continue
                block_coeff = j_coeffs[j]
                for (cr, cc), base_coeff in C_base[k].items():
                    if base_coeff == 0:
                        continue
                    i_idx = bi * 2 + cr
                    k_idx = bj * 2 + cc
                    val = block_coeff * base_coeff
                    if val != 0:
                        add_W(k_idx, i_idx, col, val)

    # Convert to lists
    rank = 49
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]

    for (row, col), val in U.items():
        U_list[row][col] = val
    for (row, col), val in V.items():
        V_list[row][col] = val
    for (row, col), val in W.items():
        W_list[row][col] = val

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
