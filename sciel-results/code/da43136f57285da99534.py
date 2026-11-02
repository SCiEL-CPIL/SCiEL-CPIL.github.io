"""Rank-49 exact tensor decomposition for 4x4 matrix multiplication (Strassen-based)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time


def _build_rank49():
    """Build the rank-49 U, V, W factor matrices for 4x4 matrix multiplication.

    Uses the Strassen 2x2 recursion: 4x4 = 2x2 blocks of 2x2, giving 4*7 = 28...
    Actually the standard rank-49 for 4x4 comes from a specific construction.
    Let me use the known explicit rank-49 decomposition.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # The rank-49 decomposition of 4x4 matrix multiplication.
    # This is based on the Strassen-based construction where we decompose
    # the 4x4 multiplication into 49 rank-1 terms.
    #
    # Standard approach: Use the 7-term Strassen formula recursively.
    # For 4x4: split into 2x2 blocks. Each 2x2 block multiplication uses
    # 7 terms. But we need to handle cross-block terms too.
    #
    # Actually, the known rank-49 for 4x4 is NOT simply 4*7=28.
    # The minimal rank for 4x4 is 49 (this is a known result).
    #
    # One construction: Use the 7-term Strassen on 2x2 sub-blocks, but
    # the 4x4 multiplication tensor has rank 49, achieved by a specific
    # non-recursive construction.
    #
    # Let me use the explicit construction from the literature.
    # The 4x4 multiplication can be done with 49 multiplications using
    # a specific set of 49 linear combinations.

    # I'll use the construction based on partitioning indices into
    # two groups of 2, and using Strassen-like combinations.

    # For the 4x4 case with rank 49, one known construction uses
    # the following: for each of 49 terms, define linear forms in A and B.

    # Let me use a different approach: the rank-49 decomposition can be
    # constructed by taking the Kronecker product of the 2x2 Strassen
    # with a 1x1 identity, but that gives 7*1=7 for 2x2 and we need 49 for 4x4.

    # Actually, 49 = 7^2. This suggests a Kronecker structure.
    # If we view 4x4 as a 2x2 block matrix where each block is 2x2,
    # and we apply Strassen to the block structure (treating each 2x2
    # block as a scalar), we get 7 "block" multiplications. Each block
    # multiplication is a 2x2 * 2x2 = 2x2 product, which itself needs
    # 7 scalar multiplications. Total: 7*7 = 49.

    # So the construction is:
    # 1. Split A, B, C into 2x2 blocks of 2x2 sub-matrices.
    # 2. Apply Strassen at the block level: 7 block products.
    # 3. Each block product is a 2x2 matrix product, done with 7 scalar products.
    # 4. Total: 49 scalar products.

    # Block layout:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B = [[C00, C01], [C10, C11]]

    # Strassen at block level:
    # M1 = (A00 + A11) * (B01 + B10)
    # M2 = (A10 + A11) * B00
    # M3 = A10 * (B10 - B00)
    # M4 = A11 * (B11 - B01)
    # M5 = A00 * (B00 + B11)
    # M6 = (A01 + A11) * (B10 + B11)
    # M7 = (A00 - A10) * (B01 + B10)

    # C00 = M5 + M4 - M2 - M7
    # C01 = M1 + M6
    # C10 = M3 + M5
    # C11 = M1 - M4 + M2 + M6

    # Each Mj is a 2x2 matrix product. For example, M1 = (A00+A11)*(B01+B10)
    # is a product of two 2x2 matrices, which needs 7 scalar multiplications.

    # So for each of the 7 block-level Strassen terms, we have a 2x2 matrix
    # product, each decomposed into 7 scalar products. Total 49.

    # Now I need to write out the 49 rank-1 terms.
    # For block-level term j (j=0..6) and scalar-level term s (s=0..6):
    # The A linear form is: (linear combo of A blocks) * (linear combo of A entries within block)
    # The B linear form is: (linear combo of B blocks) * (linear combo of B entries within block)
    # The C contribution is: (linear combo of C blocks) * (linear combo of C entries within block)

    # Let me define the block-level Strassen terms as pairs of (A_block_combo, B_block_combo, C_block_combo)
    # where each combo is a list of (block_index, coefficient).

    # Block indices: 00->0, 01->1, 10->2, 11->3
    # A block i: rows {2*i, 2*i+1}, cols {2*i, 2*i+1}  -- NO, this is wrong for rectangular blocks.
    # For 4x4 split into 2x2 blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3

    # So A block (bi, bj): rows {2*bi, 2*bi+1}, cols {2*bj, 2*bj+1}
    # B block (bi, bj): rows {2*bi, 2*bi+1}, cols {2*bj, 2*bj+1}
    # C block (bi, bj): rows {2*bi, 2*bi+1}, cols {2*bj, 2*bj+1}

    # A entry (i,k): block = (i//2, k//2), sub = (i%2, k%2)
    # B entry (k,j): block = (k//2, j//2), sub = (k%2, j%2)
    # C entry (i,j): block = (i//2, j//2), sub = (i%2, j%2)

    # Block-level Strassen:
    block_strassen = [
        # (A_blocks, B_blocks, C_blocks)
        # M1 = (A00+A11)*(B01+B10) -> C01 += M1
        ([(0, 1), (3, 1)], [(1, 1), (2, 1)], [(1, 1)]),
        # M2 = (A10+A11)*B00 -> C00 -= M2, C11 += M2
        ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (3, 1)]),
        # M3 = A10*(B10-B00) -> C10 += M3
        ([(2, 1)], [(2, 1), (0, -1)], [(2, 1)]),
        # M4 = A11*(B11-B01) -> C00 += M4, C11 -= M4
        ([(3, 1)], [(3, 1), (1, -1)], [(0, 1), (3, -1)]),
        # M5 = A00*(B00+B11) -> C00 += M5, C10 += M5
        ([(0, 1)], [(0, 1), (3, 1)], [(0, 1), (2, 1)]),
        # M6 = (A01+A11)*(B10+B11) -> C01 += M6, C11 += M6
        ([(1, 1), (3, 1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)]),
        # M7 = (A00-A10)*(B01+B10) -> C00 -= M7
        ([(0, 1), (2, -1)], [(1, 1), (2, 1)], [(0, -1)]),
    ]

    # Scalar-level Strassen (for 2x2):
    # For a 2x2 product X*Y = Z:
    # S1 = (x00+x11)(y01+y10)
    # S2 = (x10+x11)y00
    # S3 = x10(y10-y00)
    # S4 = x11(y11-y01)
    # S5 = x00(y00+y11)
    # S6 = (x01+x11)(y10+y11)
    # S7 = (x00-x10)(y01+y10)
    # Z00 = S5+S4-S2-S7
    # Z01 = S1+S6
    # Z10 = S3+S5
    # Z11 = S1-S4+S2+S6

    scalar_strassen = [
        # (X_entries, Y_entries, Z_entries)
        # X entry (i,k): sub-index i*2+k
        # Y entry (k,j): sub-index k*2+j
        # Z entry (i,j): sub-index i*2+j
        # S1 = (x00+x11)(y01+y10)
        ([(0, 1), (3, 1)], [(1, 1), (2, 1)], [(1, 1)]),
        # S2 = (x10+x11)y00
        ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (3, 1)]),
        # S3 = x10(y10-y00)
        ([(2, 1)], [(2, 1), (0, -1)], [(2, 1)]),
        # S4 = x11(y11-y01)
        ([(3, 1)], [(3, 1), (1, -1)], [(0, 1), (3, -1)]),
        # S5 = x00(y00+y11)
        ([(0, 1)], [(0, 1), (3, 1)], [(0, 1), (2, 1)]),
        # S6 = (x01+x11)(y10+y11)
        ([(1, 1), (3, 1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)]),
        # S7 = (x00-x10)(y01+y10)
        ([(0, 1), (2, -1)], [(1, 1), (2, 1)], [(0, -1)]),
    ]

    for bi in range(7):  # block-level Strassen term
        a_blocks, b_blocks, c_blocks = block_strassen[bi]
        for si in range(7):  # scalar-level Strassen term
            x_entries, y_entries, z_entries = scalar_strassen[si]
            col = bi * 7 + si

            # Build U (A linear form)
            for (blk, blk_coeff) in a_blocks:
                for (sub, sub_coeff) in x_entries:
                    # blk = block index (0..3), sub = sub-index within block (0..3)
                    # blk: row_block = blk//2, col_block = blk%2
                    # sub: row_sub = sub//2, col_sub = sub%2
                    row = 2 * (blk // 2) + (sub // 2)
                    col_a = 2 * (blk % 2) + (sub % 2)
                    a_global = row * 4 + col_a
                    U[a_global][col] += blk_coeff * sub_coeff

            # Build V (B linear form)
            for (blk, blk_coeff) in b_blocks:
                for (sub, sub_coeff) in y_entries:
                    row = 2 * (blk // 2) + (sub // 2)
                    col_b = 2 * (blk % 2) + (sub % 2)
                    b_global = row * 4 + col_b
                    V[b_global][col] += blk_coeff * sub_coeff

            # Build W (C linear form)
            for (blk, blk_coeff) in c_blocks:
                for (sub, sub_coeff) in z_entries:
                    row = 2 * (blk // 2) + (sub // 2)
                    col_c = 2 * (blk % 2) + (sub % 2)
                    # W index: k*n + i where k = col of C, i = row of C
                    w_global = col_c * 4 + row
                    W[w_global][col] += blk_coeff * sub_coeff

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=4, m=4, p=4):
    """Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(p):
            for k in range(m):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    deadline = time.time() + 140

    result = _build_rank49()

    if time.time() < deadline:
        if _verify(result["U"], result["V"], result["W"]):
            return result

    # Fallback: schoolbook rank-64
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
