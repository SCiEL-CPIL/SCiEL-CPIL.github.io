"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Use the Strassen-based decomposition of rank 56 for 2x4 * 4x8.
    # This is derived by treating the 4x8 multiplication as two independent
    # 2x4 multiplications (splitting the 4 rows of the middle dimension into
    # two groups of 2), each of which can be handled by a 2x2x2 Strassen
    # block applied to 2x2x2 sub-blocks.
    #
    # More precisely:
    # A is 2x4, B is 4x8, C is 2x8.
    # Split B into B_left (4x4) and B_right (4x4).
    # Then C_left = A * B_left (2x4), C_right = A * B_right (2x4).
    # For each 2x4 multiplication (2x2 * 2x2), we can use Strassen on
    # 2x2 blocks. A 2x4 matrix A can be seen as [A00 A01; A10 A11] where
    # each Ajk is 1x2. Similarly B_left is [B00 B01; B10 B11] where each
    # is 2x2.
    #
    # Actually, let's use the approach from inspiration 1 which already
    # has a working rank-56 decomposition. We'll use that directly.
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 decomposition patterns
    # For a 2x2 * 2x2 multiplication:
    # C[0,0] = M0 - M1 + M2 + M5
    # C[0,1] = M1 + M4
    # C[1,0] = M3 + M5
    # C[1,1] = M0 - M2 + M3 + M6
    # where:
    # M0 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M1 = (A[1,0] + A[1,1]) * B[0,0]
    # M2 = A[0,0] * (B[0,1] - B[1,1])
    # M3 = A[1,1] * (B[1,0] - B[0,0])
    # M4 = (A[0,0] + A[0,1]) * B[1,1]
    # M5 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M6 = (A[1,1] - A[0,1]) * (B[1,0] + B[1,1])
    
    # In tensor form, for each r in 0..6, we have U_r, V_r, W_r vectors
    # such that sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = C[i,k]
    
    # Let's define the 7 Strassen terms for 2x2x2:
    # Indexing: i,j,k in {0,1}
    # U has entries at (i,j), V at (j,k), W at (k,i)
    
    strassen_U = [
        [(0,0,1), (1,1,1)],      # M0: A[0,0] + A[1,1]
        [(1,0,1), (1,1,1)],      # M1: A[1,0] + A[1,1]
        [(0,0,1)],                # M2: A[0,0]
        [(1,1,1)],                # M3: A[1,1]
        [(0,0,1), (0,1,1)],      # M4: A[0,0] + A[0,1]
        [(1,0,1), (0,0,-1)],     # M5: A[1,0] - A[0,0]
        [(1,1,1), (0,1,-1)],     # M6: A[1,1] - A[0,1]
    ]
    strassen_V = [
        [(0,0,1), (1,1,1)],      # M0: B[0,0] + B[1,1]
        [(0,0,1)],                # M1: B[0,0]
        [(0,1,1), (1,1,-1)],     # M2: B[0,1] - B[1,1]
        [(1,0,1), (0,0,-1)],     # M3: B[1,0] - B[0,0]
        [(1,1,1)],                # M4: B[1,1]
        [(0,0,1), (0,1,1)],      # M5: B[0,0] + B[0,1]
        [(1,0,1), (1,1,1)],      # M6: B[1,0] + B[1,1]
    ]
    strassen_W = [
        [(0,0,1), (1,1,-1)],     # M0: contributes to C[0,0] (+), C[1,1] (-)
        [(0,0,-1), (0,1,1)],     # M1: contributes to C[0,0] (-), C[0,1] (+)
        [(0,0,1), (1,1,-1)],     # M2: contributes to C[0,0] (+), C[1,1] (-)
        [(1,0,1), (1,1,1)],      # M3: contributes to C[1,0] (+), C[1,1] (+)
        [(0,1,1)],                # M4: contributes to C[0,1] (+)
        [(0,0,1), (1,0,1)],      # M5: contributes to C[0,0] (+), C[1,0] (+)
        [(0,0,-1), (1,1,1)],     # M6: contributes to C[0,0] (-), C[1,1] (+)
    ]
    
    # Now we need to apply this to 2x4 * 4x8.
    # Split the middle dimension (size 4) into two parts of size 2.
    # Split the output dimension (size 8) into four parts of size 2.
    
    # For each pair (j_block, k_block) where j_block in {0,1} and k_block in {0,3},
    # we have a 2x2 * 2x2 sub-problem.
    
    # j_block 0: j in {0,1}, j_block 1: j in {2,3}
    # k_block 0: k in {0,1}, k_block 1: k in {2,3}, k_block 2: k in {4,5}, k_block 3: k in {6,7}
    
    # For each (j_block, k_block) pair, we use 7 Strassen terms.
    # Total: 2 * 4 * 7 = 56 terms.
    
    def add_strassen_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = idx
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = idx
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = idx
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val
    
    # Apply 8 blocks: 2 j-blocks * 4 k-blocks
    block_idx = 0
    for j_block in range(2):
        for k_block in range(4):
            j_offset = j_block * 2
            k_offset = k_block * 2
            add_strassen_block(block_idx * 7, 0, j_offset, k_offset)
            block_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
