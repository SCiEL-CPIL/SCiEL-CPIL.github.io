"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Score is rank; lower is better.
    """
    n, m, p = 4, 5, 6
    deadline = time.time() + 110  # safety margin

    # Strategy: partition into blocks and use Strassen for 2x2x2 subproblems
    # where possible, schoolbook for the rest.
    
    # Partition: A is 4x5, B is 5x6, C is 4x6
    # Split into 2x2 blocks of size (2,2) in A and B, giving (2,2) in C.
    # A: rows 0-3, cols 0-4. Split rows into [0,1] and [2,3], cols into [0,1] and [2,3].
    # B: rows 0-4, cols 0-5. Split rows into [0,1] and [2,3], cols into [0,1] and [2,3].
    # This leaves out row 4 of A/B (index 4) and col 4,5 of B.
    
    # Let me think about a cleaner partition.
    # A: 4x5. Split into 2x2 blocks: need 5 = 2+3. So rows: [0,1],[2,3]; cols: [0,1],[2,3],[4]
    # B: 5x6. Split rows: [0,1],[2,3],[4]; cols: [0,1],[2,3],[4,5]
    # C: 4x6. Rows: [0,1],[2,3]; Cols: [0,1],[2,3],[4,5]
    
    # Block structure:
    # A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :] + A[0:2, 4:5] * B[4:5, :] -> C[0:2, :]
    # A[2:4, 0:2] * B[0:2, :] + A[2:4, 2:4] * B[2:4, :] + A[2:4, 4:5] * B[4:5, :] -> C[2:4, :]
    
    # For the 2x2x2 blocks (where both A block and B block are 2x2), use Strassen (rank 7)
    # For other blocks, use schoolbook.
    
    # Let's enumerate all (i_row_block, j_col_block, k_intermediate_block) triples:
    # i in {0,1} (row blocks of A/C), j in {0,1,2} (col blocks of A / intermediate), k in {0,1,2} (row blocks of B / col blocks of C)
    
    # Block sizes:
    # A row blocks: size [2, 2]
    # A col / B row blocks: size [2, 2, 1]  (intermediate dimension m=5)
    # B col / C col blocks: size [2, 2, 2]  (p=6)
    
    # For each (i, j, k): A_block is (2 x size_j), B_block is (size_j x size_k), C_block is (2 x size_k)
    # If size_j == 2 and size_k == 2, we can use Strassen-like or other optimizations.
    # Actually Strassen requires both to be 2x2. So we need A_block 2x2, B_block 2x2, C_block 2x2.
    # That happens when j is in {0,1} (size 2) and k is in {0,1} (size 2).
    
    # Let me count:
    # j=0 (size 2), j=1 (size 2), j=2 (size 1)
    # k=0 (size 2), k=1 (size 2), k=2 (size 2)
    
    # Blocks where Strassen applies: j in {0,1}, k in {0,1} -> 4 blocks of rank 7 each = 28
    # Other blocks:
    # j=0,1 with k=2: 2 blocks, each is 2x2 * 2x2 -> C is 2x2. Schoolbook rank = 4. Total = 8
    # j=2 with k=0,1,2: 3 blocks, each is 2x1 * 1x2 -> C is 2x2. Schoolbook rank = 4. Total = 12
    
    # Total rank = 28 + 8 + 12 = 48? That seems too low. Let me reconsider.
    
    # Wait, schoolbook for 2x2 * 2x2: rank = 2*2*2 = 8. Not 4.
    # For 2x1 * 1x2: rank = 2*1*2 = 4. That's correct.
    
    # Let me recount:
    # Strassen blocks (2x2x2): 4 blocks * 7 = 28
    # Schoolbook 2x2x2 blocks: j in {0,1}, k=2: 2 blocks * 8 = 16
    # Schoolbook 2x1x2 blocks: j=2, k in {0,1,2}: 3 blocks * 4 = 12
    # Total = 28 + 16 + 12 = 56
    
    # Hmm, that's much better than 101. Let me verify this is correct.
    
    # Actually, let me reconsider the partition more carefully.
    # n=4, m=5, p=6
    # A: 4x5, B: 5x6, C: 4x6
    
    # Row partition of A (n=4): [0,1], [2,3] -> sizes [2, 2]
    # Col partition of A / Row partition of B (m=5): [0,1], [2,3], [4] -> sizes [2, 2, 1]
    # Col partition of B (p=6): [0,1], [2,3], [4,5] -> sizes [2, 2, 2]
    
    # For each (i, j, k) where i in [0,1], j in [0,1,2], k in [0,1,2]:
    # A_sub = A[2i:2i+2, col_start_j:col_start_j+size_j]  (2 x size_j)
    # B_sub = B[row_start_j:row_start_j+size_j, col_start_k:col_start_k+size_k]  (size_j x size_k)
    # C_sub = C[2i:2i+2, col_start_k:col_start_k+size_k]  (2 x size_k)
    
    # The product A_sub @ B_sub contributes to C_sub.
    
    # If size_j == 2 and size_k == 2: 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
    # If size_j == 2 and size_k == 2 but we're not using Strassen: schoolbook rank 8.
    # If size_j == 1 and size_k == 2: 2x1 * 1x2 -> 2x2. Schoolbook rank 4.
    
    # j=0: size 2, j=1: size 2, j=2: size 1
    # k=0: size 2, k=1: size 2, k=2: size 2
    
    # Strassen eligible: (j,k) in {(0,0),(0,1),(1,0),(1,1)} -> 4 per i, 2 i's = 8 blocks? No.
    # For each (i,j,k), it's one block. i in {0,1}, so for each (j,k) pair, there are 2 blocks.
    # Strassen: (j,k) in {(0,0),(0,1),(1,0),(1,1)}, i in {0,1} -> 4*2 = 8 blocks of rank 7 = 56
    # Schoolbook 2x2x2: (j,k) = {(0,2),(1,2)}, i in {0,1} -> 2*2 = 4 blocks of rank 8 = 32
    # Schoolbook 2x1x2: (j,k) = {(2,0),(2,1),(2,2)}, i in {0,1} -> 3*2 = 6 blocks of rank 4 = 24
    
    # Total = 56 + 32 + 24 = 112. Still better than 120.
    
    # Can I do better? Let me try a different partition.
    # What if I use 1x1 blocks in some places to enable more Strassen-like tricks?
    
    # Actually, let me just implement the 112 rank solution first and verify it works.
    # Then I can try to optimize further.
    
    # Build U, V, W matrices
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # 20 x 112
    V = [[0] * rank for _ in range(m * p)]  # 30 x 112
    W = [[0] * rank for _ in range(p * n)]  # 24 x 112
    
    col_offset = 0
    
    # Block boundaries
    a_row_starts = [0, 2]  # n=4, 2 blocks of size 2
    a_col_starts = [0, 2, 4]  # m=5, sizes [2,2,1]
    b_col_starts = [0, 2, 4]  # p=6, sizes [2,2,2]
    
    a_row_sizes = [2, 2]
    a_col_sizes = [2, 2, 1]
    b_col_sizes = [2, 2, 2]
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a_r0 = a_row_starts[i]
                a_c0 = a_col_starts[j]
                b_r0 = a_col_starts[j]  # B rows align with A cols
                b_c0 = b_col_starts[k]
                
                a_rows = [a_r0, a_r0 + 1]
                a_cols = list(range(a_c0, a_c0 + a_col_sizes[j]))
                b_rows = list(range(b_r0, b_r0 + a_col_sizes[j]))
                b_cols = list(range(b_c0, b_c0 + b_col_sizes[k]))
                c_rows = a_rows
                c_cols = b_cols
                
                if a_col_sizes[j] == 2 and b_col_sizes[k] == 2:
                    # Strassen 2x2x2, rank 7
                    # Local indices: a_idx = r*2+c for r,c in {0,1}
                    # b_idx = r*2+c for r,c in {0,1}
                    # c_idx = k*n + i for k,i in {0,1} (local)
                    
                    # Strassen formulas:
                    # M1 = (a00 + a11)(b00 + b11)
                    # M2 = (a10 + a11)b00
                    # M3 = a00(b01 - b11)
                    # M4 = a11(b10 - b00)
                    # M5 = (a00 + a01)b11
                    # M6 = (a10 - a00)(b00 + b01)
                    # M7 = (a01 - a11)(b10 + b11)
                    #
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # For each M_t, define u, v, w vectors of length 4 (for a, b) and 4 (for c)
                    # u is 4-dim (a00, a01, a10, a11)
                    # v is 4-dim (b00, b01, b10, b11)
                    # w is 4-dim (c00, c01, c10, c11) where c_idx = c_col*2 + c_row? 
                    # Wait, C is 4x6, so C[r, c] with r in 0..3, c in 0..5.
                    # Local c_idx for the 2x2 block: rows [a_r0, a_r0+1], cols [b_c0, b_c0+1]
                    # Global W index: c_global_col * n + c_global_row = c_global_col * 4 + c_global_row
                    
                    # Let me define local to global mappings
                    # a local (lr, lc) -> global (a_r0 + lr, a_c0 + lc) -> U index: (a_r0+lr)*m + (a_c0+lc)
                    # b local (lr, lc) -> global (b_r0 + lr, b_c0 + lc) -> V index: (b_r0+lr)*p + (b_c0+lc)
                    # c local (lr, lc) -> global (a_r0 + lr, b_c0 + lc) -> W index: (b_c0+lc)*n + (a_r0+lr)
                    
                    strassen_terms = [
                        # (u_coeffs, v_coeffs, w_coeffs)
                        # M1
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        # M2
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
                        # M3
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
                        # M4
                        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, -1]),
                        # M5
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        # M6
                        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
                        # M7
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                    ]
                    
                    for (u_c, v_c, w_c) in strassen_terms:
                        col = col_offset
                        # u: 4 entries for a local indices
                        for lr in range(2):
                            for lc in range(2):
                                local_idx = lr * 2 + lc
                                global_row = a_r0 + lr
                                global_col = a_c0 + lc
                                U[global_row * m + global_col][col] = u_c[local_idx]
                        # v: 4 entries for b local indices
                        for lr in range(2):
                            for lc in range(2):
                                local_idx = lr * 2 + lc
                                global_row = b_r0 + lr
                                global_col = b_c0 + lc
                                V[global_row * p + global_col][col] = v_c[local_idx]
                        # w: 4 entries for c local indices
                        for lr in range(2):
                            for lc in range(2):
                                local_idx = lr * 2 + lc
                                global_row = a_r0 + lr
                                global_col = b_c0 + lc
                                W[global_col * n + global_row][col] = w_c[local_idx]
                        col_offset += 1
                    
                elif a_col_sizes[j] == 2 and b_col_sizes[k] == 2:
                    pass  # Already handled above
                else:
                    # Schoolbook
                    # a is 2 x size_j, b is size_j x size_k, c is 2 x size_k
                    for lr in range(2):
                        for lc in range(len(a_cols)):
                            for lk in range(len(b_cols)):
                                col = col_offset
                                global_a_row = a_r0 + lr
                                global_a_col = a_c0 + lc
                                global_b_row = b_r0 + lc
                                global_b_col = b_c0 + lk
                                global_c_row = a_r0 + lr
                                global_c_col = b_c0 + lk
                                U[global_a_row * m + global_a_col][col] = 1
                                V[global_b_row * p + global_b_col][col] = 1
                                W[global_c_col * n + global_c_row][col] = 1
                                col_offset += 1
    
    # Verify col_offset matches rank
    assert col_offset == rank, f"Rank mismatch: expected {rank}, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
