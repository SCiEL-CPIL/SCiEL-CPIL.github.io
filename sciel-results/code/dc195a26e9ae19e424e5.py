"""Kronecker-product decomposition for 4x4x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Rank-7 Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # C[i,j] = sum_k P_k * W[i,j,k]
    # P_k = (sum_a U_a,i * A_a) * (sum_b V_b,k * B_b)  -- but we need to be careful with indices

    # Standard Strassen for C = A * B where A, B, C are 2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 - P3 - P4 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P5 - P7

    # U1: 4x7, rows indexed by A's elements in row-major: [A00, A01, A10, A11]
    # Each column k corresponds to the coefficients of A in P_k
    U1 = np.array([
        [ 1,  0,  1,  0,  1,  1,  0],  # A00
        [ 0,  0,  0,  0,  1,  0, -1],  # A01
        [ 0,  1,  0,  0,  0, -1,  0],  # A10
        [ 1,  1,  0,  1,  0,  0,  1],  # A11
    ], dtype=float)

    # V1: 4x7, rows indexed by B's elements in row-major: [B00, B01, B10, B11]
    V1 = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],  # B00
        [ 0,  0,  1,  0,  0,  1,  1],  # B01
        [ 0,  0,  0,  1,  0,  0,  1],  # B10
        [ 1,  0, -1,  0,  1,  0,  0],  # B11
    ], dtype=float)

    # W1: 4x7, rows indexed by C's elements in the trace convention W[k*2+i, r]
    # For 2x2 output, p=2, n=2, so W has p*n = 4 rows
    # Row index = k*n + i where C[i,k] is the element
    # C[0,0] -> row 0*2+0 = 0
    # C[0,1] -> row 1*2+0 = 2
    # C[1,0] -> row 0*2+1 = 1
    # C[1,1] -> row 1*2+1 = 3
    # So rows are: [C00, C10, C01, C11] in terms of (k,i) pairs: (0,0),(0,1),(1,0),(1,1)
    # Wait: W[k*n + i, r] means for output C[i, k], the row is k*n + i.
    # C[0,0]: k=0, i=0 -> row 0
    # C[0,1]: k=1, i=0 -> row 2
    # C[1,0]: k=0, i=1 -> row 1
    # C[1,1]: k=1, i=1 -> row 3
    # So W1 rows: row0=C00, row1=C10, row2=C01, row3=C11

    W1 = np.array([
        [ 1,  0, -1, -1,  0,  1,  0],  # C00 = P1 - P3 - P4 + P6
        [ 0,  1,  0,  1,  0,  0,  0],  # C10 = P2 + P4
        [ 0,  0,  1,  0,  1,  0,  0],  # C01 = P3 + P5
        [ 1, -1, 0, -1, 1,  0, -1],  # C11 = P1 - P2 + P5 - P7
    ], dtype=float)

    # Rank-11 decomposition for 2x2x3 (A: 2x2, B: 2x3, C: 2x3)
    # This is a known decomposition. Let's construct it.
    # A is 2x2: [A00, A01; A10, A11]
    # B is 2x3: [B00, B01, B02; B10, B11, B12]
    # C is 2x3: [C00, C01, C02; C10, C11, C12]
    #
    # One approach: split B into B[:, 0:2] and B[:, 2:3]
    # For B[:, 0:2]: use Strassen-like (rank 7)
    # For B[:, 2:3]: use schoolbook (rank 4) but maybe shareable
    # Total: 7 + 4 = 11

    # Let's define the rank-11 decomposition explicitly.
    # First 7 terms: Strassen for columns 0,1 of B
    # U2_first7: same as U1 but we only use A's elements
    # V2_first7: coefficients for B[:, 0:2]
    # W2_first7: coefficients for C[:, 0:2]

    # For the first 7 terms (Strassen on B columns 0,1):
    # P1 = (A00+A11)(B00+B10) -> C00
    # P2 = (A10+A11)B00 -> C10
    # P3 = A00(B01-B11) -> C01
    # P4 = A11(B10-B00) -> C10
    # P5 = (A00+A01)B11 -> C01
    # P6 = (A10-A00)(B00+B01) -> C00, C01
    # P7 = (A01-A11)(B10+B11) -> C11

    # U2a = U1 (same A coefficients)
    # V2a: 6x7, rows for B elements in row-major: [B00, B01, B02, B10, B11, B12]
    # For columns 0,1: B00 (row 0), B01 (row 1), B10 (row 3), B11 (row 4)
    V2a = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],  # B00
        [ 0,  0,  1,  0,  0,  1,  1],  # B01
        [ 0,  0,  0,  0,  0,  0,  0],  # B02 (unused in first 7)
        [ 0,  0,  0,  1,  0,  0,  1],  # B10
        [ 1,  0, -1,  0,  1,  0,  0],  # B11
        [ 0,  0,  0,  0,  0,  0,  0],  # B12 (unused in first 7)
    ], dtype=float)

    # W2a: 6x7, rows for C elements in trace convention W[k*2+i, r]
    # C is 2x3, so p=3, n=2, total 6 rows
    # C[0,0]: k=0,i=0 -> row 0
    # C[0,1]: k=1,i=0 -> row 2
    # C[0,2]: k=2,i=0 -> row 4
    # C[1,0]: k=0,i=1 -> row 1
    # C[1,1]: k=1,i=1 -> row 3
    # C[1,2]: k=2,i=1 -> row 5
    # For first 7 terms, only C[:, 0:2] are affected
    W2a = np.array([
        [ 1,  0, -1, -1,  0,  1,  0],  # C00
        [ 0,  1,  0,  1,  0,  0,  0],  # C10
        [ 0,  0,  1,  0,  1,  0,  0],  # C01
        [ 1, -1, 0, -1, 1,  0, -1],  # C11
        [ 0,  0,  0,  0,  0,  0,  0],  # C02
        [ 0,  0,  0,  0,  0,  0,  0],  # C12
    ], dtype=float)

    # Last 4 terms: schoolbook for column 2 of B
    # P8 = A00 * B02 -> C02
    # P9 = A01 * B12 -> C02
    # P10 = A10 * B02 -> C12
    # P11 = A11 * B12 -> C12

    U2b = np.array([
        [ 1,  0,  0,  0],  # A00
        [ 0,  1,  0,  0],  # A01
        [ 0,  0,  1,  0],  # A10
        [ 0,  0,  0,  1],  # A11
    ], dtype=float)

    V2b = np.array([
        [ 1,  0,  0,  0],  # B00
        [ 0,  0,  0,  0],  # B01
        [ 0,  0,  1,  0],  # B02
        [ 0,  0,  0,  0],  # B10
        [ 0,  0,  0,  0],  # B11
        [ 0,  1,  0,  1],  # B12
    ], dtype=float)

    W2b = np.array([
        [ 0,  0,  0,  0],  # C00
        [ 0,  0,  0,  0],  # C10
        [ 0,  0,  0,  0],  # C01
        [ 0,  0,  0,  0],  # C11
        [ 1,  1,  0,  0],  # C02
        [ 0,  0, 1,  1],  # C12
    ], dtype=float)

    # Combine to get rank-11 decomposition for 2x2x3
    U2 = np.hstack([U1, U2b])  # 4x11
    V2 = np.hstack([V2a, V2b])  # 6x11
    W2 = np.hstack([W2a, W2b])  # 6x11

    # Now we need to build the 4x4x6 decomposition using Kronecker product structure.
    # A is 4x4, B is 4x6, C is 4x6.
    # We can view A as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B is 4x6. We can view B as having 2x2 blocks of size 2x3.
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x3.
    # C is 4x6. C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x3.
    #
    # C_ij = sum_{kl} A_ik * B_kj
    # This is a 2x2 matrix multiplication at the block level, where each "entry" is a 2x3 matrix.
    # But our Strassen decomposition works for 2x2x2 (all 2x2).
    # And our 2x2x3 decomposition works for 2x2 A, 2x2x3 B, 2x2x3 C.
    #
    # The Kronecker product approach:
    # We have a 2x2x2 Strassen (rank 7) and a 2x2x3 decomposition (rank 11).
    # The total rank should be 7 * 11 = 77.
    #
    # U for 4x4x6: shape (16, 77)
    # V for 4x4x6: shape (24, 77)
    # W for 4x4x6: shape (24, 77)
    #
    # The Kronecker product structure:
    # For each pair (k1, k2) where k1 in 0..6 (Strassen index) and k2 in 0..10 (2x2x3 index),
    # we create a term in the decomposition.
    #
    # U[i4, (k1, k2)] where i4 is the flattened index of A's 4x4 elements.
    # A's elements in row-major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ..., A[3,3]
    # Block A_ik where i in {0,1}, k in {0,1}, each is 2x2.
    # A_00 = A[0:2, 0:2], A_01 = A[0:2, 2:4], A_10 = A[2:4, 0:2], A_11 = A[2:4, 2:4]
    #
    # For the Kronecker product, U[i4, (k1, k2)] = U1[i_block, k1] * U2[i_sub, k2]
    # where i4 = i_block * 4 + i_sub (roughly, depends on ordering)
    #
    # Let me think about this more carefully.
    #
    # Actually, the correct way: the tensor T[i, j, k] = delta (matrix mult tensor)
    # can be decomposed as sum over r of U[i, r] * V[j, r] * W[k, r]
    #
    # For the Kronecker product of two tensor decompositions:
    # If T1 has decomposition (U1, V1, W1) with rank r1, and T2 has (U2, V2, W2) with rank r2,
    # then T1 x T2 (Kronecker product of tensors) has decomposition (U1 x U2, V1 x V2, W1 x W2) with rank r1 * r2.
    #
    # Here, we're not exactly doing a Kronecker product of two tensor decompositions in the usual sense.
    # Instead, we're exploiting the block structure.
    #
    # Let me reconsider. The 4x4x6 multiplication tensor can be seen as:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # If we partition A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x3, C into 2x2 blocks of 2x3,
    # then C_block[a, b] = sum_c A_block[a, c] * B_block[c, b]
    # where A_block is 2x2 of 2x2, B_block is 2x2 of 2x3, C_block is 2x2 of 2x3.
    #
    # This is a "block matrix multiplication" where the block sizes are 2x2, 2x3, 2x3.
    #
    # The rank of this block multiplication can be bounded by the product of the ranks of:
    # - The 2x2x2 Strassen (for the block-level 2x2 multiplication)
    # - The 2x2x3 decomposition (for the element-level 2x2x3 multiplication)
    #
    # So total rank = 7 * 11 = 77.
    #
    # Now let's construct the factors.
    #
    # U has shape (16, 77). Row index i4 in 0..15 corresponds to A[i4 // 4, i4 % 4].
    # Block index: block_row = i4 // 4 // 2, block_col = i4 // 4 % 2 ... wait
    # A is 4x4. Row-major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ...
    # Block A_ab where a in {0,1}, b in {0,1}: A[2a:2a+2, 2b:2b+2]
    # A[0,0] is in block A_00, A[0,1] is in block A_00, A[0,2] is in block A_01, A[0,3] is in block A_01
    # A[1,0] is in block A_00, A[1,1] is in block A_00, A[1,2] is in block A_01, A[1,3] is in block A_01
    # A[2,0] is in block A_10, A[2,1] is in block A_10, A[2,2] is in block A_11, A[2,3] is in block A_11
    # A[3,0] is in block A_10, A[3,1] is in block A_10, A[3,2] is in block A_11, A[3,3] is in block A_11
    #
    # So for A[i, j]: block_row = i // 2, block_col = j // 2, sub_row = i % 2, sub_col = j % 2
    # i4 = i * 4 + j
    # block_row = (i4 // 4) // 2 = i4 // 8
    # block_col = (i4 // 4) % 2 = (i4 // 4) % 2
    # sub_row = i4 % 4 // 2 ... let me just compute directly.

    rank = 7 * 11
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((24, rank), dtype=float)

    for k1 in range(7):
        for k2 in range(11):
            col = k1 * 11 + k2
            # U[i4, col] = U1[block_a, k1] * U2[sub_idx, k2]
            # where block_a = block index in A (0..3 for 2x2 blocks), sub_idx = sub-block index (0..3 for 2x2)
            for i4 in range(16):
                i = i4 // 4
                j = i4 % 4
                block_row = i // 2
                block_col = j // 2
                sub_row = i % 2
                sub_col = j % 2
                block_idx = block_row * 2 + block_col  # 0,1,2,3 for A_00, A_01, A_10, A_11
                sub_idx = sub_row * 2 + sub_col  # 0,1,2,3 for sub-block positions
                U[i4, col] = U1[block_idx, k1] * U2[sub_idx, k2]

            # V[j4, col] = V1[block_c, k1] * V2[sub_idx2, k2]
            # B is 4x6. B[j, k] where j in 0..3, k in 0..5.
            # Block structure: B has 2x2 blocks of 2x3.
            # B[j, k]: block_row = j // 2, block_col = k // 3, sub_row = j % 2, sub_col = k % 3
            # j4 = j * 6 + k
            for j4 in range(24):
                j = j4 // 6
                k = j4 % 6
                block_row = j // 2
                block_col = k // 3
                sub_row = j % 2
                sub_col = k % 3
                block_idx = block_row * 2 + block_col  # 0,1,2,3 for B_00, B_01, B_10, B_11
                # For 2x2x3, V2 has 6 rows: B elements in row-major of 2x3: [B00, B01, B02, B10, B11, B12]
                # sub_idx2 = sub_row * 3 + sub_col
                sub_idx2 = sub_row * 3 + sub_col
                V[j4, col] = V1[block_idx, k1] * V2[sub_idx2, k2]

            # W[k4, col] = W1[block_d, k1] * W2[sub_idx3, k2]
            # C is 4x6. W trace convention: W[k*n + i, r] where C[i, k].
            # p=6, n=4. W row index = k * 4 + i where C[i, k].
            # C[i, k]: block_row = i // 2, block_col = k // 3, sub_row = i % 2, sub_col = k % 3
            # W row = k * 4 + i
            for wrow in range(24):
                k = wrow // 4
                i = wrow % 4
                block_row = i // 2
                block_col = k // 3
                sub_row = i % 2
                sub_col = k % 3
                block_idx = block_row * 2 + block_col  # 0,1,2,3 for C_00, C_01, C_10, C_11
                # For 2x2x3, W2 has 6 rows: C elements in trace convention W[k*2+i, r] for 2x3 C
                # C is 2x3, p=3, n=2. W row = k * 2 + i where C[i, k].
                # sub_idx3 = sub_col * 2 + sub_row  (since k corresponds to sub_col, i to sub_row)
                sub_idx3 = sub_col * 2 + sub_row
                W[wrow, col] = W1[block_idx, k1] * W2[sub_idx3, k2]

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
