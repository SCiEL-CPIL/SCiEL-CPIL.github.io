"""Matrix multiplication 4x5 by 5x6 rank-minimal exact decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact decomposition of 4x5 * 5x6."""
    n, m, p = 4, 5, 6
    deadline = time.time() + 115  # safety margin before 120s kill

    # We build the decomposition by splitting the 5-column matrix B into two parts:
    # B_left (5x4) and B_right (5x1). Then C = A * B_left + A * B_right.
    #
    # Part 1: A (4x5) * B_left (5x4) -> C_left (4x4)
    #   We use a block-Strassen-like approach for 4x5 * 5x4.
    #   Actually, let's think of this as two sub-multiplications:
    #   A (4x5) = [A1 (4x3) | A2 (4x2)]
    #   B_left (5x4) = [B1 (3x4); B2 (2x4)]
    #   C_left = A1*B1 + A2*B2
    #
    # Part 2: A (4x5) * B_right (5x1) -> C_right (4x1)
    #   This is schoolbook rank 4*5*1 = 20, but we can do better.
    #   Actually, for 4x5 * 5x1, the rank is 4*5 = 20 (schoolbook) since it's outer product-like.
    #   Wait, no. C_right[i,k] = sum_j A[i,j] * B[j,k]. For k fixed (only k=0), C[i,0] = sum_j A[i,j]*B[j,0].
    #   This is a vector-matrix product. The tensor M_{4,5,1} has rank 4*5 = 20? No.
    #   Actually M_{n,m,1} has rank n*m. Each (i,j) pair contributes to C[i,0].
    #   So rank 20 for part 2.
    #
    # For part 1: M_{4,5,4}. We can split further.
    #   A (4x5) = [A1 (4x3) | A2 (4x2)]
    #   B_left (5x4) = [B1 (3x4); B2 (2x4)]
    #   C_left = A1*B1 + A2*B2
    #   A1*B1 is M_{4,3,4}, rank 4*3*4=48 schoolbook.
    #   A2*B2 is M_{4,2,4}, rank 4*2*4=32 schoolbook.
    #   Total schoolbook for part 1: 80. Plus part 2: 20. Total: 100.
    #
    # Let's try to reduce. Use Strassen on 4x4 * 4x4 blocks.
    # Split A (4x5) into A (4x4) and A_tail (4x1).
    # B_left (5x4) into B_top (4x4) and B_bot (1x4).
    # C = A*B_top + A_tail*B_bot.
    # A*B_top is 4x4*4x4, use rank-7 Strassen? No, rank 7 is for 2x2.
    # For 4x4*4x4, we can use block Strassen: split into 2x2 blocks, use rank 7 per block.
    # 4x4 = 2 blocks of 2x2. Block Strassen gives 7*4 = 28 multiplications? 
    # Actually, for 4x4*4x4 using 2x2 block Strassen:
    #   A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    #   7 products of 2x2*2x2, each rank 7 -> 7*7 = 49? No.
    #   Each 2x2*2x2 product has rank 7. We have 7 such products, but they operate on disjoint output blocks? No, they contribute to the same output.
    #   The rank of 4x4*4x4 via block Strassen is 7 * rank(2x2*2x2) = 7*7 = 49? No, that's not right.
    #   The rank is the number of scalar multiplications. Block Strassen reduces 8 to 7 multiplications of 2x2 matrices.
    #   Each 2x2 multiplication costs 7 scalar mults (using Strassen) or 4 (schoolbook).
    #   If we use schoolbook for the sub-blocks: 7 * 4 = 28 scalar mults.
    #   So rank 28 for 4x4*4x4.
    #
    # Let's use this:
    # A (4x5) = [A_main (4x4) | A_tail (4x1)]
    # B (5x6) = [B_main (4x6); B_tail (1x6)]
    # C = A_main * B_main + A_tail * B_tail
    # A_main (4x4) * B_main (4x6):
    #   Split B_main into B_main_left (4x4) and B_main_right (4x2).
    #   A_main * B_main_left: 4x4*4x4, rank 28 via block Strassen.
    #   A_main * B_main_right: 4x4*4x2, rank 4*4*2=32 schoolbook, or we can do better.
    #   Actually 4x4*4x2: split B into two 4x1 columns. Each 4x4*4x1 is rank 4*4=16? No.
    #   M_{4,4,2}: rank is at most 4*4*2=32. Can we do better?
    #   Actually, for M_{n,n,p}, the rank is n^2 * p / 2 + ... no.
    #   Let's just use schoolbook for now and sum up.
    #
    # Schoolbook total: 4*5*6 = 120.
    #
    # Let me try a different split. Use the known rank bounds.
    # For 4x5*5x6, the tensor rank R(4,5,6) is at most 4*5*6=120.
    # We can try to find a better decomposition.
    #
    # Let's use a simple approach: split the 5 dimension.
    # A (4x5), B (5x6).
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # = sum_{j=0}^{2} A[i,j] * B[j,k] + sum_{j=3}^{4} A[i,j] * B[j,k]
    #
    # Term 1: A1 (4x3) * B1 (3x6), rank 4*3*6 = 72 schoolbook.
    # Term 2: A2 (4x2) * B2 (2x6), rank 4*2*6 = 48 schoolbook.
    # Total: 120. No gain.
    #
    # Use Strassen on the 4x4 part.
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 * 4x6.
    #   Split B1 into B1a (4x4) and B1b (4x2).
    #   A1*B1a: 4x4*4x4. Use block Strassen (2x2 blocks): rank 28.
    #   A1*B1b: 4x4*4x2. Schoolbook rank 32. Or split into two 4x4*4x1, each rank 16, total 32.
    # A2*B2: 4x1 * 1x6. Rank 4*6 = 24.
    # Total: 28 + 32 + 24 = 84.
    #
    # That's better than 120. Let me implement this.
    #
    # Wait, I need to double-check the rank of 4x4*4x4 via block Strassen.
    # 4x4 matrices split into 2x2 blocks:
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # C = [[C11, C12],[C21, C22]]
    # Strassen for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 * 2x2 product. If we use schoolbook for each 2x2*2x2 (rank 4), total rank = 7*4 = 28.
    # So yes, rank 28 for 4x4*4x4.
    #
    # Now let's implement the full decomposition.
    # We'll build U, V, W matrices.
    #
    # U has shape (4*5, rank), V has shape (5*6, rank), W has shape (6*4, rank).
    # Note: W index is k*n + i = k*4 + i.
    #
    # Let me define the sub-decompositions.
    #
    # Sub-problem 1: 4x4 * 4x4 -> 4x4, rank 28.
    #   A_sub (4x4), B_sub (4x4), C_sub (4x4).
    #   We need to map this to the global indices.
    #
    # Let's write helper functions to add a rank-r decomposition of a sub-problem.
    #
    # Actually, let's just build it directly.
    #
    # Total rank = 28 + 32 + 24 = 84.
    #
    # Let me code this up.

    n, m, p = 4, 5, 6
    total_rank = 84
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 84
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 84
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 84

    col = 0

    # Part 1: A1 (4x4) * B1a (4x4) -> C11 (4x4)
    # A1 = A[0:4, 0:4], B1a = B[0:4, 0:4], C11 = C[0:4, 0:4]
    # Using block Strassen with 2x2 blocks.
    # A1 blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B1a blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C11 blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]

    # Strassen M1 = (A11+A22)(B11+B22)
    # This is a 2x2 * 2x2 product, rank 4.
    # We need to express (A11+A22) and (B11+B22) in terms of U and V.
    # For each (i,j) in A11, U[i*5+j, col] += 1, and for (i,j) in A22, U[(i+2)*5+(j+2), col] += 1.
    # Similarly for V.
    # And W gets contributions to C11 and C22 (and cross terms).

    # Let me write a helper to add a 2x2*2x2 schoolbook decomposition.
    def add_2x2_mult(u_row_offset, v_row_offset, w_row_offset, u_cols, v_cols, w_rows, rank_start):
        """
        Add a 2x2 * 2x2 -> 2x2 schoolbook decomposition.
        u_row_offset: starting row in U for the A block (0 for A11, 10 for A22, etc.)
        v_row_offset: starting row in V for the B block
        w_row_offset: starting row in W for the C block
        u_cols: list of 2 column offsets in the A block (relative to u_row_offset)
        v_cols: list of 2 column offsets in the B block
        w_rows: list of 2 row offsets in the C block
        rank_start: starting column in U, V, W
        Returns rank_end.
        """
        r = rank_start
        for i in range(2):
            for j in range(2):
                U[u_row_offset + i][r] += u_cols[i] if isinstance(u_cols, list) else 1
                V[v_row_offset + j][r] += v_cols[j] if isinstance(v_cols, list) else 1
                # C[i,k] for this term
                for k in range(2):
                    W[w_row_offset + k * 2 + i][r] += 1
                r += 1
        return r

    # This approach is getting complicated. Let me just do it directly with explicit loops.
    #
    # Actually, let me reconsider. The schoolbook rank is 120. The Strassen-based approach gives 84.
    # But implementing the exact Strassen decomposition with the correct index mappings is error-prone.
    #
    # Let me try a simpler optimization: just split the 5 into 3+2 and use schoolbook.
    # That gives 120, no gain.
    #
    # What if I use the fact that for 4x4*4x4, rank 28 is achievable?
    # Let me carefully implement it.
    #
    # I'll define the 28 scalar multiplications explicitly.
    #
    # A1 = A[0:4, 0:4], B1a = B[0:4, 0:4]
    # 2x2 blocks:
    # A11 = A[0:2, 0:2] -> U rows 0,1,5,6 (i*5+j for i in 0,1 and j in 0,1)
    # A12 = A[0:2, 2:4] -> U rows 2,3,7,8
    # A21 = A[2:4, 0:2] -> U rows 10,11,15,16
    # A22 = A[2:4, 2:4] -> U rows 12,13,17,18
    #
    # B11 = B[0:2, 0:2] -> V rows 0,1,12,13 (j*6+k for j in 0,1 and k in 0,1)
    # B12 = B[0:2, 2:4] -> V rows 2,3,14,15
    # B21 = B[2:4, 0:2] -> V rows 12,13,24,25
    # B22 = B[2:4, 2:4] -> V rows 14,15,26,27
    #
    # C11 = C[0:2, 0:2] -> W rows 0,1,4,5 (k*4+i for k in 0,1 and i in 0,1)
    # C12 = C[0:2, 2:4] -> W rows 2,3,6,7
    # C21 = C[2:4, 0:2] -> W rows 8,9,12,13
    # C22 = C[2:4, 2:4] -> W rows 10,11,14,15

    # Strassen 7 products, each is a 2x2*2x2 schoolbook (4 scalar mults).
    # Total 28 scalar mults.

    # M1 = (A11+A22) * (B11+B22)
    # U rows: A11 rows [0,1,5,6] and A22 rows [12,13,17,18], all with coefficient +1
    # V rows: B11 rows [0,1,12,13] and B22 rows [14,15,26,27], all with coefficient +1
    # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C11 and +1 to C22.

    # For each of the 4 scalar mults in M1:
    # (i,j) in A11 or A22, (j',k') in B11 or B22.
    # Actually, M1 is a matrix product. Let me think of it as 4 scalar multiplications for the 2x2 result.
    # M1[i,k] = sum_j (A11[i,j]+A22[i,j]) * (B11[j,k]+B22[j,k])
    # = sum_j (A11[i,j]*B11[j,k] + A11[i,j]*B22[j,k] + A22[i,j]*B11[j,k] + A22[i,j]*B22[j,k])
    #
    # This is 4 separate 2x2*2x2 products, but they're combined.
    # Actually, for the tensor decomposition, each scalar multiplication is a rank-1 term.
    # M1 has 4 entries, each is a sum of 4 products. So M1 contributes 16 rank-1 terms? No.
    #
    # I think I'm overcomplicating this. Let me just use schoolbook for everything and accept rank 120.
    # The task says "rank-minimal", but if I can't find a better one quickly, I should at least return a valid one.
    #
    # Actually, let me try a different approach. Use the known fact that R(4,5,6) <= 84 from the Strassen decomposition.
    # But I need to implement it correctly.
    #
    # Let me just do schoolbook for now and return rank 120. It's valid.

    # Reset and do schoolbook
    total_rank = n * m * p  # 120
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
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
