import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    # Schoolbook rank is 105
    # We will use Strassen on 2x2 blocks where possible.
    # A is 3x5, B is 5x7.
    # We can partition A into blocks of size 2x2, 2x1, 1x2, 1x1.
    # We can partition B into blocks of size 2x2, 2x1, 1x2, 1x1.
    # To apply Strassen, we need 2x2 blocks in both A and B (relative to their dimensions).
    # Let's partition rows of A into [2, 1] and cols of A (and rows of B) into [2, 3] or [2, 2, 1].
    # Let's try partitioning:
    # A rows: i in [0, 1, 2] -> [0,1], [2]
    # A cols: j in [0, 1, 2, 3, 4] -> [0,1], [2,3,4]
    # B rows: j in [0, 1, 2, 3, 4] -> [0,1], [2,3,4]
    # B cols: k in [0, 1, 2, 3, 4, 5, 6] -> [0,1], [2,3,4,5,6]
    
    # This creates 4x4 = 16 blocks.
    # Block (0,0) of A is 2x2. Block (0,0) of B is 2x2. -> Can use Strassen (7 mults).
    # Block (0,1) of A is 2x3. Block (0,1) of B is 2x5. -> No.
    # Block (1,0) of A is 1x2. Block (1,0) of B is 1x2. -> No (need 2x2).
    # Block (1,1) of A is 1x3. Block (1,1) of B is 1x5. -> No.
    
    # Only the top-left 2x2 corner of A and top-left 2x2 corner of B form a valid 2x2x2 block for Strassen.
    # Wait, Strassen applies to multiplying a 2x2 matrix by a 2x2 matrix.
    # Here A_block is 2x2 (rows 0,1; cols 0,1) and B_block is 2x2 (rows 0,1; cols 0,1).
    # The result C_block will be 2x2 (rows 0,1; cols 0,1).
    # This affects C[0:2, 0:2].
    
    # Are there other 2x2 blocks?
    # A rows [0,1], cols [0,1] is 2x2.
    # A rows [0,1], cols [1,2] is 2x2.
    # We can partition A cols as [0], [1,2], [3,4].
    # A rows [0,1] is 2x1, 2x2, 2x2.
    # Let's try to maximize the number of 2x2x2 blocks.
    
    # Partition A rows: [0,1], [2] (sizes 2, 1)
    # Partition A cols: [0,1], [2,3], [4] (sizes 2, 2, 1)
    # Partition B cols: [0,1], [2,3], [4,5,6] (sizes 2, 2, 3)
    # B rows must match A cols partition.
    
    # Blocks of A:
    # (0,0): 2x2 (rows 0,1; cols 0,1)
    # (0,1): 2x2 (rows 0,1; cols 2,3)
    # (0,2): 2x1 (rows 0,1; cols 4)
    # (1,0): 1x2 (rows 2; cols 0,1)
    # (1,1): 1x2 (rows 2; cols 2,3)
    # (1,2): 1x1 (rows 2; cols 4)
    
    # Blocks of B:
    # (0,0): 2x2 (rows 0,1; cols 0,1)
    # (0,1): 2x2 (rows 0,1; cols 2,3)
    # (0,2): 2x3 (rows 0,1; cols 4,5,6)
    # (1,0): 2x2 (rows 2,3; cols 0,1)
    # (1,1): 2x2 (rows 2,3; cols 2,3)
    # (1,2): 2x3 (rows 2,3; cols 4,5,6)
    # (2,0): 1x2 (rows 4; cols 0,1)
    # (2,1): 1x2 (rows 4; cols 2,3)
    # (2,2): 1x3 (rows 4; cols 4,5,6)
    
    # Where can we apply Strassen?
    # We need A_block and B_block to be compatible for matrix multiplication?
    # No, Strassen is for C = A B.
    # If we partition A into blocks A_ij and B into blocks B_jk, then C_ik = sum_j A_ij B_jk.
    # Strassen can be applied to any term A_ij B_jk if both are 2x2 matrices?
    # No, Strassen is an algorithm to compute the product of two 2x2 matrices using 7 multiplications.
    # It requires A_ij to be 2x2 and B_jk to be 2x2.
    # Wait, A_ij has dimensions (rows_i) x (cols_j). B_jk has dimensions (rows_j) x (cols_k).
    # For A_ij B_jk to be a standard matrix product, cols_j must equal rows_j.
    # Here, the "cols_j" of A are the indices of the columns of A, which are the same as the "rows_j" of B.
    # So if we partition the shared index (size m=5) into chunks of size 2, 2, 1.
    # Then A_00 is 2x2, A_01 is 2x2, A_02 is 2x1.
    # B_00 is 2x2, B_01 is 2x2, B_02 is 2x3.
    # B_10 is 2x2, B_11 is 2x2, B_12 is 2x3.
    # B_20 is 1x2, B_21 is 1x2, B_22 is 1x3.
    
    # Terms A_ij B_jk:
    # j=0 (size 2):
    #   A_00 (2x2) * B_00 (2x2) -> Strassen (7)
    #   A_00 (2x2) * B_01 (2x2) -> Strassen (7)
    #   A_00 (2x2) * B_02 (2x3) -> Cannot use standard 2x2 Strassen directly?
    #      We can pad B_02 to 2x4? Or use a different algorithm?
    #      If we use schoolbook for 2x2 * 2x3, cost is 2*2*3 = 12.
    #      Can we do better?
    #      2x2 * 2x3 is equivalent to 6 scalar multiplications? No, it's 12 scalar mults in schoolbook.
    #      Rank of 2x2x3 matrix multiplication?
    #      It is known that M(2,2,3) has rank <= 2 * M(2,2,1)? No.
    #      Actually, we can just use schoolbook for the non-2x2x2 blocks.
    #   A_10 (1x2) * B_00 (2x2) -> Schoolbook (1*2*2=4)
    #   A_10 (1x2) * B_01 (2x2) -> Schoolbook (4)
    #   A_10 (1x2) * B_02 (2x3) -> Schoolbook (1*2*3=6)
    #   A_20? No, A only has 2 row blocks [0,1], [2]. So A_20 doesn't exist.
    #   Wait, A rows are [0,1] and [2]. So i=0,1.
    #   A_00, A_01, A_02.
    #   A_10, A_11, A_12.
    
    # Let's list all terms A_ij B_jk and their costs.
    # Partitions:
    # A rows: I0=[0,1], I1=[2]
    # Shared index (cols of A, rows of B): J0=[0,1], J1=[2,3], J2=[4]
    # B cols: K0=[0,1], K1=[2,3], K2=[4,5,6]
    
    # Blocks:
    # A_00: 2x2
    # A_01: 2x2
    # A_02: 2x1
    # A_10: 1x2
    # A_11: 1x2
    # A_12: 1x1
    
    # B_00: 2x2
    # B_01: 2x2
    # B_02: 2x3
    # B_10: 2x2
    # B_11: 2x2
    # B_12: 2x3
    # B_20: 1x2
    # B_21: 1x2
    # B_22: 1x3
    
    # Terms:
    # i=0 (A rows 0,1):
    #   j=0 (shared 0,1):
    #     k=0 (B cols 0,1): A_00 * B_00. 2x2 * 2x2. Strassen (7).
    #     k=1 (B cols 2,3): A_00 * B_01. 2x2 * 2x2. Strassen (7).
    #     k=2 (B cols 4,5,6): A_00 * B_02. 2x2 * 2x3. Schoolbook (12).
    #   j=1 (shared 2,3):
    #     k=0: A_01 * B_10. 2x2 * 2x2. Strassen (7).
    #     k=1: A_01 * B_11. 2x2 * 2x2. Strassen (7).
    #     k=2: A_01 * B_12. 2x2 * 2x3. Schoolbook (12).
    #   j=2 (shared 4):
    #     k=0: A_02 * B_20. 2x1 * 1x2. Schoolbook (4).
    #     k=1: A_02 * B_21. 2x1 * 1x2. Schoolbook (4).
    #     k=2: A_02 * B_22. 2x1 * 1x3. Schoolbook (6).
    
    # i=1 (A rows 2):
    #   j=0:
    #     k=0: A_10 * B_00. 1x2 * 2x2. Schoolbook (4).
    #     k=1: A_10 * B_01. 1x2 * 2x2. Schoolbook (4).
    #     k=2: A_10 * B_02. 1x2 * 2x3. Schoolbook (6).
    #   j=1:
    #     k=0: A_11 * B_10. 1x2 * 2x2. Schoolbook (4).
    #     k=1: A_11 * B_11. 1x2 * 2x2. Schoolbook (4).
    #     k=2: A_11 * B_12. 1x2 * 2x3. Schoolbook (6).
    #   j=2:
    #     k=0: A_12 * B_20. 1x1 * 1x2. Schoolbook (2).
    #     k=1: A_12 * B_21. 1x1 * 1x2. Schoolbook (2).
    #     k=2: A_12 * B_22. 1x1 * 1x3. Schoolbook (3).
    
    # Total rank:
    # Strassen: 4 * 7 = 28
    # Schoolbook:
    #   12 + 12 = 24
    #   4 + 4 + 6 = 14
    #   4 + 4 + 6 = 14
    #   4 + 4 + 6 = 14
    #   2 + 2 + 3 = 7
    # Sum schoolbook: 24 + 14 + 14 + 14 + 7 = 73
    # Total: 28 + 73 = 101.
    
    # Previous best was 99.
    # The previous program used 6 Strassen blocks.
    # How did they get 6?
    # Maybe they partitioned differently.
    # If we partition shared index into 2, 2, 1, we got 4 Strassen blocks.
    # What if we partition A rows into 1, 1, 1? No, need 2.
    # What if we partition A rows into 2, 1 and B cols into 2, 2, 3?
    # We need A_ij and B_jk to be 2x2.
    # This requires A rows chunk size 2, B cols chunk size 2, and shared chunk size 2.
    # So we need chunks of size 2 in all three dimensions.
    # n=3, m=5, p=7.
    # We can fit at most one chunk of size 2 in n (3 = 2+1).
    # We can fit at most two chunks of size 2 in m (5 = 2+2+1).
    # We can fit at most three chunks of size 2 in p (7 = 2+2+2+1).
    
    # The number of Strassen blocks is (num 2-chunks in n) * (num 2-chunks in m) * (num 2-chunks in p).
    # 1 * 2 * 3 = 6 Strassen blocks.
    # This matches the "6 Strassen blocks" finding!
    
    # Let's verify this partition.
    # n (rows of A): [0,1], [2] -> 1 chunk of size 2.
    # m (cols of A / rows of B): [0,1], [2,3], [4] -> 2 chunks of size 2.
    # p (cols of B): [0,1], [2,3], [4,5], [6] -> 3 chunks of size 2.
    
    # Strassen blocks are at (i,j,k) where i,j,k correspond to 2-chunks.
    # i=0 (rows 0,1), j in {0,1} (cols 0,1 and 2,3), k in {0,1,2} (cols 0,1, 2,3, 4,5).
    # Total 1 * 2 * 3 = 6 blocks.
    # Each block is 2x2x2. Rank 7.
    # Total Strassen rank = 6 * 7 = 42.
    
    # Remaining terms are schoolbook.
    # Total terms = 105.
    # Strassen terms cover:
    # i=0 (2 rows), j=0 (2 cols), k=0 (2 cols) -> 2*2*2 = 8 scalar products.
    # 6 blocks * 8 = 48 scalar products covered.
    # Remaining scalar products = 105 - 48 = 57.
    # Each remaining term is a schoolbook product A_ij B_jk (scalar level? No, block level).
    # Wait, the "rank" counts the number of scalar multiplications in the decomposition.
    # In the seed program, each scalar product (i,j,k) is one rank.
    # In Strassen, 8 scalar products are replaced by 7 scalar products.
    # So we save 1 rank per Strassen block.
    # 6 blocks -> save 6 ranks.
    # 105 - 6 = 99.
    
    # So the strategy is:
    # 1. Identify the 6 2x2x2 sub-tensors.
    # 2. Replace each with a Strassen decomposition (rank 7).
    # 3. Keep all other (i,j,k) as schoolbook (rank 1).
    
    # Implementation:
    # We need to map the Strassen indices to the global (i,j,k) indices.
    # For each Strassen block, we define the local indices (a,b,c) in {0,1}^3.
    # The global indices are:
    # i = n_offset[i_idx] + a
    # j = m_offset[j_idx] + b
    # k = p_offset[k_idx] + c
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A11 - A10)(B01 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M7
    
    # We need to construct U, V, W for these 7 terms for each block.
    # Then fill the rest with schoolbook.
    
    n, m, p = 3, 5, 7
    # Offsets for 2-chunks
    n_offsets = [0]  # [0,1]
    m_offsets = [0, 2]  # [0,1], [2,3]
    p_offsets = [0, 2, 4]  # [0,1], [2,3], [4,5]
    
    # Strassen coefficients
    # Term: (U_coeffs, V_coeffs, W_coeffs)
    # Local indices: a,b in {0,1} for A, c in {0,1} for B?
    # A is 2x2, indices (a,b). B is 2x2, indices (b,c)? No, B is 2x2, indices (j,k).
    # In the sub-block, A is 2x2 (rows a, cols b), B is 2x2 (rows b, cols c).
    # Wait, the shared index is the middle one.
    # So A_block is 2x2 (rows i', cols j'), B_block is 2x2 (rows j', cols k').
    # The Strassen formulas apply to these 2x2 matrices.
    
    # Let's define the Strassen terms in terms of (i', j', k').
    # M1: (i'=0, j'=0) + (i'=1, j'=1) in U; (j'=0, k'=0) + (j'=1, k'=1) in V; (k'=0, i'=0) + (k'=1, i'=1) in W?
    # No, W is indexed by (k, i).
    # C[i', k'] = sum_j' A[i', j'] B[j', k'].
    # Strassen computes C' = A' B'.
    # The contributions to C'[i', k'] are the W coefficients.
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: (0,0):1, (1,1):1
    # V: (0,0):1, (1,1):1
    # W: Contributes to C00 (1), C11 (1).
    # W[0,0] = 1, W[1,1] = 1.
    
    # M2 = (A10 + A11) B00
    # U: (1,0):1, (1,1):1
    # V: (0,0):1
    # W: Contributes to C10 (1), C11 (-1).
    # W[0,1] = 1, W[1,1] = -1.
    
    # M3 = A00 (B10 - B11)
    # U: (0,0):1
    # V: (1,0):1, (1,1):-1
    # W: Contributes to C01 (1), C11 (1).
    # W[1,0] = 1, W[1,1] = 1.
    
    # M4 = A11 (B01 - B00)
    # U: (1,1):1
    # V: (0,1):1, (0,0):-1
    # W: Contributes to C00 (1), C10 (1).
    # W[0,0] = 1, W[0,1] = 1.
    
    # M5 = (A00 + A10) B11
    # U: (0,0):1, (1,0):1
    # V: (1,1):1
    # W: Contributes to C00 (-1), C01 (1).
    # W[0,0] = -1, W[1,0] = 1.
    
    # M6 = (A10 - A00)(B00 + B10)
    # U: (1,0):1, (0,0):-1
    # V: (0,0):1, (1,0):1
    # W: Contributes to C11 (1).
    # W[1,1] = 1.
    
    # M7 = (A11 - A10)(B01 + B11)
    # U: (1,1):1, (1,0):-1
    # V: (0,1):1, (1,1):1
    # W: Contributes to C00 (1).
    # W[0,0] = 1.
    
    # We will store these as lists of (index, coeff).
    # U index is i'*2 + j'? No, global index is n_offset + i', m_offset + j'.
    # Let's just store local offsets and add to global.
    
    strassen_terms = [
        {
            "U": [(0, 1), (1, 1)],
            "V": [(0, 1), (1, 1)],
            "W": [(0, 1), (1, 1)] # (k', i') -> (0,0), (1,1)
        },
        {
            "U": [(1, 1), (1, 1)],
            "V": [(0, 1)],
            "W": [(0, 1), (1, -1)] # (0,1), (1,1)
        },
        {
            "U": [(0, 1)],
            "V": [(1, 1), (1, -1)],
            "W": [(1, 1), (1, 1)] # (1,0), (1,1)
        },
        {
            "U": [(1, 1)],
            "V": [(0, 1), (0, -1)],
            "W": [(0, 1), (0, 1)] # (0,0), (0,1)
        },
        {
            "U": [(0, 1), (1, 1)],
            "V": [(1, 1)],
            "W": [(0, -1), (1, 1)] # (0,0), (1,0)
        },
        {
            "U": [(1, 1), (0, -1)],
            "V": [(0, 1), (1, 1)],
            "W": [(1, 1)] # (1,1)
        },
        {
            "U": [(1, 1), (1, -1)],
            "V": [(0, 1), (1, 1)],
            "W": [(0, 1)] # (0,0)
        }
    ]
    
    # Note: W index in local block is (k', i').
    # Global W index is p_offset[k_idx] + k', n_offset[i_idx] + i'.
    # But W is a matrix of size p*n.
    # The task says: W[k*n+i, r].
    # So global W index is (p_offset + k') * n + (n_offset + i').
    
    # Initialize U, V, W
    # We don't know the rank yet.
    # Rank = 105 - 6 = 99.
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_used = set()
    
    # Add Strassen blocks
    for i_idx, n_off in enumerate(n_offsets):
        for j_idx, m_off in enumerate(m_offsets):
            for k_idx, p_off in enumerate(p_offsets):
                # Check if this is a 2x2x2 block
                # It is, by construction of offsets.
                for t in range(7):
                    term = strassen_terms[t]
                    for u_idx, coeff in term["U"]:
                        # u_idx is (i', j')? No, in my dict I used (i', j')?
                        # Let's check M1: U: (0,0), (1,1). Yes, (i', j').
                        i_prime, j_prime = u_idx
                        u_global = (n_off + i_prime) * m + (m_off + j_prime)
                        U[u_global][col] = coeff
                    
                    for v_idx, coeff in term["V"]:
                        # v_idx is (j', k')
                        j_prime, k_prime = v_idx
                        v_global = (m_off + j_prime) * p + (p_off + k_prime)
                        V[v_global][col] = coeff
                    
                    for w_idx, coeff in term["W"]:
                        # w_idx is (k', i')
                        k_prime, i_prime = w_idx
                        w_global = (p_off + k_prime) * n + (n_off + i_prime)
                        W[w_global][col] = coeff
                    
                    col += 1
                    strassen_used.add((n_off, m_off, p_off, t)) # Dummy
    
    # Add Schoolbook for the rest
    # We need to iterate all i, j, k
    # and skip those covered by Strassen.
    # Strassen covers:
    # i in [n_off, n_off+1], j in [m_off, m_off+1], k in [p_off, p_off+1]
    # for each offset.
    
    def is_in_strassen(i, j, k):
        for n_off in n_offsets:
            if i >= n_off and i < n_off + 2:
                for m_off in m_offsets:
                    if j >= m_off and j < m_off + 2:
                        for p_off in p_offsets:
                            if k >= p_off and k < p_off + 2:
                                return True
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not is_in_strassen(i, j, k):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
