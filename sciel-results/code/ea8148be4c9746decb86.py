"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Strassen rank-7 decomposition for 2x2 matrices.
    # Indices: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]B[k,j]
    # We represent each 2x2 matrix as a 4-vector: [m00, m01, m10, m11]
    # The 7 multiplications are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B00)
    # M4 = A11 (B01 - B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # Define U, V, W vectors for each of the 7 terms
    # U[i] corresponds to row i of A (flattened as 2x2 -> 4 elements: 00,01,10,11)
    # V[j] corresponds to row j of B (flattened as 2x2 -> 4 elements: 00,01,10,11)
    # W[k] corresponds to output C positions (flattened as 2x2 -> 4 elements: 00,10,01,11 in trace convention)
    # Wait, the trace convention for output is W[k*n+i, r] where the output index is k*n + i.
    # For a 2x2 block at position (row_block, col_block), the local indices are i in {0,1}, k in {0,1}.
    # The global output index would be (row_block*2 + i) * (num_col_blocks*2) + (col_block*2 + k)? 
    # No, the standard matrix multiplication output C has shape n x p.
    # The tensor is T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product.
    # Actually, let's just build it directly for the 4x4x6 case.

    # Let's use a simpler approach: 
    # 4x4x6 can be seen as a tensor network.
    # We can use the fact that 4x4x6 = (2x2x2) x (2x2x2) x (1x1x3) or similar.
    # But the directive says: "construct a rank-77 decomposition for 4x4x6 by applying a tensor network contraction that maps the 4x4x6 problem to a parallel composition of four independent 2x2x2 Strassen instances"
    # Wait, 4x4x6. If we split into 2x2 blocks:
    # A is 4x4 -> 2x2 blocks of 2x2
    # B is 4x6 -> 2x3 blocks of 2x2 (last column is 2x2)
    # So it's like a 2x2 grid of blocks, but B has 3 columns of blocks.
    # So we have 2*3 = 6 block multiplications? No, that's not right.
    # A[i_block, j_block] * B[j_block, k_block] -> C[i_block, k_block]
    # i_block in {0,1}, j_block in {0,1}, k_block in {0,1,2}
    # So for each (i_block, k_block) pair, we have a 2x2x2 multiplication.
    # There are 2*3 = 6 such independent 2x2x2 multiplications.
    # Each has rank 7, so total rank = 6*7 = 42.
    # But the directive says rank-77. Why 77?
    # Maybe it's not independent. Let's re-read.
    # "construct a rank-77 decomposition ... parallel composition of four independent 2x2x2 Strassen instances"
    # Four instances? 4*7 = 28, not 77.
    # Maybe it's 11 instances? 11*7 = 77.
    # Or maybe it's not just Strassen.
    # Let's just implement the block-diagonal approach with 6 independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 6 * 7 = 42.
    # This is much better than 96.

    # Let's define the 7 Strassen terms for a single 2x2x2 multiplication.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [1, 0, -1, 0]
    # M7: [0, 1, 0, -1]

    # V vectors (for B):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 0, 1, -1]  # b10 - b00
    # M4: [0, 1, 0, -1]  # b01 - b11
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]

    # W vectors (for C):
    # C00 = M1 + M4 - M5 + M7 -> [1, 0, 0, 0] for M1, [1, 0, 0, 0] for M4, [-1, 0, 0, 0] for M5, [1, 0, 0, 0] for M7
    # Wait, W vectors are for the output.
    # For M1, it contributes to C00 and C11.
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, -1]  # C10 += M2, C11 -= M2
    # M3: [0, 0, 1, 1]   # C01 += M3, C11 += M3
    # M4: [1, 0, 0, 0]   # C00 += M4
    # M5: [-1, 0, 1, 0]  # C00 -= M5, C01 += M5
    # M6: [0, 1, 0, 1]   # C10 += M6, C11 += M6
    # M7: [1, 0, 0, 0]   # C00 += M7

    # Let's verify:
    # C00 = M1 + M4 - M5 + M7 = 1 + 1 - (-1) + 1 = 1+1+1+1 = 4? No.
    # M1: C00 += 1
    # M4: C00 += 1
    # M5: C00 -= 1
    # M7: C00 += 1
    # Sum = 1 + 1 - 1 + 1 = 2? But it should be 1.
    # Let's re-check the Strassen formulas.
    # C00 = M1 - M5 + M4 + M7
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M4 = A11(B01-B11) = A11B01 - A11B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    # C00 = M1 - M5 + M4 + M7 = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B11 + A01B11) + (A11B01 - A11B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A11B01 - A11B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 + A11B01 + A01B10 - A11B10 - A11B11
    # This is not C00 = A00B00 + A01B10.
    # I must have the Strassen formulas wrong.
    # Let's look up the correct Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # Let's re-verify C00:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M5 = -A00B11 - A01B11
    # M4 = A11B01 - A11B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    # Sum = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A11B01 - A11B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 + A11B01 + A01B10 - A11B10 - A11B11
    # Still not C00.
    # I think the standard Strassen is for 2x2, but maybe I have the indices wrong.
    # Let's just use the schoolbook for 2x2x2, which has rank 4, and then combine.
    # No, the task is to find a rank-minimal decomposition. The known rank for 2x2x2 is 7.
    # Let's just implement the 6 independent 2x2x2 multiplications with schoolbook (rank 4 each) to get rank 24, which is better than 96.
    # Then we can try to use Strassen to get rank 42.

    # Let's implement the 6 independent 2x2x2 multiplications with schoolbook.
    # A is 4x4, B is 4x6, C is 4x6.
    # Block (i, j) of A is 2x2, block (j, k) of B is 2x2.
    # i in {0, 1}, j in {0, 1}, k in {0, 1, 2}.
    # For each (i, k), we have a 2x2x2 multiplication over j in {0, 1}.
    # Wait, that's not right.
    # C[i_block, k_block] = sum_{j_block} A[i_block, j_block] * B[j_block, k_block]
    # So for each (i_block, k_block), we have a 2x2 matrix multiplication where the "scalar" field is the 2x2 blocks.
    # No, that's not right either.
    # A[i_block, j_block] is a 2x2 matrix. B[j_block, k_block] is a 2x2 matrix.
    # So C[i_block, k_block] = sum_{j_block=0}^{1} A[i_block, j_block] * B[j_block, k_block]
    # This is a 2x2 matrix multiplication where the "scalars" are 2x2 matrices.
    # So it's a 2x2x2 tensor multiplication where each "scalar" is a 2x2 matrix.
    # This can be decomposed into 2*2*2 = 8 independent 2x2 matrix multiplications?
    # No, it's a single 2x2x2 multiplication over the 2x2 matrix ring.
    # The rank of this is at most 2*2*2 * 4 = 32? No.
    # Let's just do the schoolbook for the whole 4x4x6 multiplication, which has rank 96.
    # Then we can try to use Strassen on the 2x2 blocks.

    # Let's just implement the schoolbook decomposition for 4x4x6.
    rank = 4 * 4 * 6
    U = [[0] * rank for _ in range(4 * 4)]
    V = [[0] * rank for _ in range(4 * 6)]
    W = [[0] * rank for _ in range(6 * 4)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j][column] = 1
                V[j * 6 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
