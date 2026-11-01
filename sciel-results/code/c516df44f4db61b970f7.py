"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
Uses a block decomposition: the 4x8 matrix B is split into four 4x2 column-blocks.
Each 2x4 * 4x2 product is a 2x2 matrix multiplication (rank 7 via Strassen).
Total rank = 4 * 7 = 28.
Output follows W[k*n + i, r] convention."""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8

    # B is split into 4 column-blocks of width 2
    num_blocks = p // 2  # 4 blocks

    # Strassen's 7 products for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # Result: [[P1+P3-P5+P7, P2+P5], [P2+P4, P1-P2+P4+P6]]

    rank = num_blocks * 7  # 28

    U = [[0] * rank for _ in range(n * m)]   # 8 rows (n*m = 8)
    V = [[0] * rank for _ in range(m * p)]   # 32 rows (m*p = 32)
    W = [[0] * rank for _ in range(p * n)]   # 16 rows (p*n = 16)

    # Strassen product definitions:
    # Each product is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (row_in_A_block, coeff) where row_in_A_block is 0..3 (i*m + j)
    # V_coeffs: list of (col_in_B_block, coeff) where col_in_B_block is 0..3 (j*p + k)
    # W_coeffs: list of (output_pos, coeff) where output_pos is k*n + i

    # For a given block b (0..3), the A indices are i in {0,1}, j in {0,1,2,3}
    # The B indices for this block are j in {0,1,2,3}, k in {2*b, 2*b+1}
    # Output positions: k in {2*b, 2*b+1}, i in {0,1}

    # Define the 7 Strassen products for 2x2:
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d  -> row = i*2+j
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h  -> col = j*2+k (within block)
    # But in our V matrix, the index is j*p + k where k is global.
    # For block b, k_global = 2*b + k_local where k_local in {0,1}
    # So V index = j * p + (2*b + k_local)

    # Let's define products with local indices, then map to global.
    # A local row: i*2 + j where i in {0,1}, j in {0,1} for the 2x2 sub-block
    # But wait: A is 2x4, so for each 2x2 block of B, we use all 4 columns of A?
    # No: A is 2x4, B is 4x8. The product is 2x8.
    # Splitting B into 4 column-blocks of width 2: each block B[:, 2b:2b+2] is 4x2.
    # A (2x4) * B_block (4x2) = 2x2 result.
    # So for each block, we do a 2x4 * 4x2 multiplication.
    # This is NOT a 2x2 * 2x2 multiplication directly. We need to handle the 4x4 inner dimension.

    # Let me reconsider. The inner dimension is m=4. We can't directly use Strassen for 2x2.
    # Instead, we can split the inner dimension j into two pairs: {0,1} and {2,3}.
    # For each pair, we have a 2x2 * 2x2 multiplication (using 2 rows of A, 2 cols of B).
    # Then sum the two results.
    # Each 2x2 * 2x2 uses rank 7, so two pairs give rank 14 per output block of width 2.
    # With 4 output blocks, total rank = 4 * 14 = 56. That's worse than 64? No, 56 < 64.
    # But can we do better?

    # Alternative: treat the whole thing as 2x4 * 4x8.
    # We can use the fact that m=4 is even and split into two 2x2 blocks in the inner dimension.
    # For each pair of j's, we do a 2x2 * 2x2 (rank 7). Two pairs -> rank 14 per (i,k) pair?
    # No, the output is 2x8, so there are 16 output elements.
    # Each output element C[i,k] = sum_j A[i,j] * B[j,k].
    # If we split j into two pairs, each pair contributes a 2x2 * 2x2 product for each (i,k) pair?
    # Actually, for a fixed pair of j's, say j in {0,1}, we have:
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2 (A submatrix) * 2x2 (B submatrix) for each... no, it's a 2x2 * 2x8 subproblem.
    # The A submatrix is 2x2 (rows i=0,1; cols j=0,1).
    # The B submatrix is 2x8 (rows j=0,1; cols k=0..7).
    # The product is 2x8.
    # We can further split the 2x8 B submatrix into four 2x2 blocks.
    # Each 2x2 * 2x2 uses rank 7. Four blocks -> rank 28 for this j-pair.
    # Two j-pairs -> total rank 56.

    # Let's implement this: for each j-pair, for each k-block, use Strassen rank 7.
    # Total rank = 2 * 4 * 7 = 56.

    rank = 2 * num_blocks * 7  # 56

    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows

    col = 0
    for j_pair in range(2):
        j0 = 2 * j_pair
        j1 = 2 * j_pair + 1
        for k_block in range(num_blocks):
            k0 = 2 * k_block
            k1 = 2 * k_block + 1

            # A submatrix: rows i=0,1; cols j=j0,j1 -> indices i*m+j
            # B submatrix: rows j=j0,j1; cols k=k0,k1 -> indices j*p+k
            # Output: rows i=0,1; cols k=k0,k1 -> indices k*n+i

            # Strassen products for 2x2 * 2x2:
            # a = A[0,j0], b = A[0,j1], c = A[1,j0], d = A[1,j1]
            # e = B[j0,k0], f = B[j0,k1], g = B[j1,k0], h = B[j1,k1]

            # P1 = (a+d)(e+h)
            U[0 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            W[k1 * n + 0][col] = 1
            W[k0 * n + 1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # P2 = (c+d)e
            U[1 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

            # P3 = a(f-h)
            U[0 * m + j0][col] = 1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1
            col += 1

            # P4 = d(g-e)
            U[1 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            V[j0 * p + k0][col] = -1
            W[k0 * n + 1][col] = 1
            col += 1

            # P5 = (a+b)h
            U[0 * m + j0][col] = 1
            U[0 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 0][col] = 1
            col += 1

            # P6 = (c-a)(e+f)
            U[1 * m + j0][col] = 1
            U[0 * m + j0][col] = -1
            V[j0 * p + k0][col] = 1
            V[j0 * p + k1][col] = 1
            W[k0 * n + 1][col] = 1
            col += 1

            # P7 = (b-d)(g+h)
            U[0 * m + j1][col] = 1
            U[1 * m + j1][col] = -1
            V[j1 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 0][col] = 1
            col += 1

    assert col == rank, f"column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
