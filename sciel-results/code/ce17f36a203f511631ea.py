import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use 3x3 block decomposition: divide 6x6 into 2x2 blocks of size 3x3.
    # Apply Strassen's algorithm (7 terms) to the 2x2 block matrix.
    # Each of the 7 Strassen multiplications is a 3x3 matrix multiplication.
    # For the 3x3 multiplications, use the trivial algorithm (rank 27).
    # Total rank = 7 * 27 = 189.
    #
    # This is equivalent to the 2x2 block approach (3x3 blocks of size 2x2, Strassen on blocks,
    # trivial 2x2 matmul) which also gives rank 3*3*3*7 = 189.
    # Both approaches yield the same rank and score = 216/189 ≈ 1.1429.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 7  # 8 * 7 = 56... wait, that's wrong.
    # Actually, for each block position (I,J,K), we need 7 Strassen terms.
    # Number of block positions: 2 * 2 * 2 = 8.
    # Total rank = 8 * 7 = 56? No, that's not right either.
    # 
    # Wait, I'm confusing myself. Let me think again.
    # 
    # In the 2x2 block approach:
    # - We have 3x3 = 9 block positions (I, J, K each in range 3)
    # - For each block position, we apply Strassen (7 terms) to a 2x2 block matrix
    # - Each Strassen term involves multiplying 2x2 matrices (trivial, rank 4)
    # - Total rank = 9 * 7 * 4 = 252? No, that's not right.
    #
    # Actually, the tensor decomposition works differently. Let me re-derive.
    #
    # The matrix multiplication tensor for n×m × m×p → n×p has rank at most n*m*p (trivial).
    # For 6×6 × 6×6 → 6×6, trivial rank = 216.
    #
    # With 2×2 blocks (block_size=2, blocks_per_dim=3):
    # - U has shape (6*6, rank) = (36, rank)
    # - V has shape (6*6, rank) = (36, rank)  
    # - W has shape (6*6, rank) = (36, rank)
    #
    # For each (I, J, K) in range(3), we apply Strassen to the 2x2 block at that position.
    # Strassen has 7 terms. Each term contributes to U, V, W.
    # Total rank = 3 * 3 * 3 * 7 = 189.
    #
    # With 3×3 blocks (block_size=3, blocks_per_dim=2):
    # - For each (I, J, K) in range(2), we apply Strassen to the 2x2 block at that position.
    # - Each Strassen term involves multiplying 3x3 matrices.
    # - If we use trivial 3x3 matmul (rank 27), each Strassen term expands to 27 rank-1 terms.
    # - Total rank = 2 * 2 * 2 * 7 * 27 = 1512? That's way more than 189.
    #
    # I see the issue. The tensor decomposition doesn't work that way. Let me re-examine.
    #
    # Actually, I think the correct way to think about it is:
    # The matrix multiplication tensor T = sum_{i,j,k} e_{i,k} ⊗ e_{j,i} ⊗ e_{j,k} (in some indexing)
    # has a decomposition where each rank-1 term is u ⊗ v ⊗ w.
    #
    # With block Strassen, we're exploiting the structure. The rank is determined by
    # the number of scalar multiplications needed.
    #
    # For 2×2 blocks: 3×3×3 = 27 block positions, each needs 7 Strassen multiplications.
    # But each "multiplication" in Strassen is of 2×2 matrices, which is 4 scalar mults.
    # However, in the tensor decomposition, each Strassen term is ONE rank-1 term
    # (because the linear combinations are absorbed into U, V, W).
    # So total rank = 27 * 7 = 189. ✓
    #
    # For 3×3 blocks: 2×2×2 = 8 block positions, each needs 7 Strassen multiplications.
    # Each "multiplication" in Strassen is of 3×3 matrices.
    # In the tensor decomposition, each Strassen term would need to be expanded into
    # 27 rank-1 terms (for the 3×3 matmul).
    # So total rank = 8 * 7 * 27 = 1512. ✗ (much worse)
    #
    # So the 2×2 block approach is better! Rank 189 vs 1512.
    #
    # The directive's claim of rank 63 is impossible. Let's stick with rank 189.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)(B00) -> C10 += 1, C11 += -1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: (A00)(B01-B11) -> C01 += 1, C11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: (A11)(B10-B00) -> C00 += 1, C10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)(B11) -> C00 += -1, C01 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
