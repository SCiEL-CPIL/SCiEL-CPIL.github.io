"""Tensor decomposition for 4x4 matrix multiplication using Strassen's algorithm."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    
    This uses the tensor square of Strassen's 2x2 algorithm. We decompose the
    4x4 matrices into 2x2 blocks and apply Strassen's algorithm at two levels.
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # all 4
    
    # We'll build the decomposition by expressing the 4x4 multiplication
    # as a composition of Strassen steps.
    
    # For 4x4, we can view matrices as 2x2 blocks of 2x2 submatrices.
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # where each Ai* is 2x2.
    
    # Strassen's algorithm for 2x2 block multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Result:
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Now, each of these 2x2 multiplications can itself be done with Strassen.
    # So we have 7 Strassen multiplications at the block level, each of which
    # requires 7 Strassen multiplications at the sub-block level.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Let's define the index mappings.
    # For a 4x4 matrix, element (i, j) has i in {0,1,2,3}, j in {0,1,2,3}.
    # Block structure: A[i,j] where i = 2*i_block + i_sub, j = 2*j_block + j_sub
    
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)  
    # W has shape (p*n, rank) = (16, 49)
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the Strassen coefficients for 2x2 block level
    # Each term r in 0..6 corresponds to one Strassen multiplication
    # For each block-level term, we have 7 sub-block-level terms
    
    # Block-level Strassen terms:
    # Term 0: (A00 + A11) * (B00 + B11)
    # Term 1: (A10 + A11) * B00
    # Term 2: A00 * (B01 - B11)
    # Term 3: A11 * (B10 - B00)
    # Term 4: (A00 + A01) * B11
    # Term 5: (A10 - A00) * (B00 + B10)
    # Term 6: (A01 - A11) * (B10 + B11)
    
    # For each block-level term t, we define:
    # - The A-blocks involved and their coefficients
    # - The B-blocks involved and their coefficients
    
    # A-block definitions for each term:
    # Term 0: A00 (+1), A11 (+1)
    # Term 1: A10 (+1), A11 (+1)
    # Term 2: A00 (+1)
    # Term 3: A11 (+1)
    # Term 4: A00 (+1), A01 (+1)
    # Term 5: A10 (+1), A00 (-1)
    # Term 6: A01 (+1), A11 (-1)
    
    a_blocks = [
        [(0, 0, 1), (1, 1, 1)],  # Term 0
        [(1, 0, 1), (1, 1, 1)],  # Term 1
        [(0, 0, 1)],             # Term 2
        [(1, 1, 1)],             # Term 3
        [(0, 0, 1), (0, 1, 1)],  # Term 4
        [(1, 0, 1), (0, 0, -1)], # Term 5
        [(0, 1, 1), (1, 1, -1)], # Term 6
    ]
    
    # B-block definitions for each term:
    # Term 0: B00 (+1), B11 (+1)
    # Term 1: B00 (+1)
    # Term 2: B01 (+1), B11 (-1)
    # Term 3: B10 (+1), B00 (-1)
    # Term 4: B11 (+1)
    # Term 5: B00 (+1), B10 (+1)
    # Term 6: B10 (+1), B11 (+1)
    
    b_blocks = [
        [(0, 0, 1), (1, 1, 1)],  # Term 0
        [(0, 0, 1)],             # Term 1
        [(0, 1, 1), (1, 1, -1)], # Term 2
        [(1, 0, 1), (0, 0, -1)], # Term 3
        [(1, 1, 1)],             # Term 4
        [(0, 0, 1), (1, 0, 1)],  # Term 5
        [(1, 0, 1), (1, 1, 1)],  # Term 6
    ]
    
    # For each block-level term, we apply Strassen to the resulting 2x2
    # multiplication. The 2x2 matrices are formed by the linear combinations
    # of the 2x2 sub-blocks.
    
    # For a 2x2 matrix formed from blocks, the Strassen decomposition
    # gives 7 rank-1 terms.
    
    # Let's define the sub-block Strassen terms.
    # For a 2x2 matrix X = [[X00, X01], [X10, X11]] and Y = [[Y00, Y01], [Y10, Y11]],
    # where each Xij and Yij are scalars (since we're at the sub-block level),
    # the Strassen terms are:
    # S0 = (X00 + X11) * (Y00 + Y11)
    # S1 = (X10 + X11) * Y00
    # S2 = X00 * (Y01 - Y11)
    # S3 = X11 * (Y10 - Y00)
    # S4 = (X00 + X01) * Y11
    # S5 = (X10 - X00) * (Y00 + Y10)
    # S6 = (X01 - X11) * (Y10 + Y11)
    
    # Result:
    # Z00 = S0 + S3 - S4 + S5
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S6
    
    # Now, for each block-level term t, we need to:
    # 1. Form the 2x2 matrix X from the A-blocks
    # 2. Form the 2x2 matrix Y from the B-blocks
    # 3. Apply Strassen to X * Y
    # 4. For each of the 7 Strassen terms, add contributions to U, V, W
    
    # The 2x2 matrix X has entries that are linear combinations of A's 2x2 sub-blocks.
    # Each A sub-block is a 2x2 matrix with elements A[2*bi + si, 2*bj + sj]
    # where bi, bj are block indices and si, sj are sub-indices.
    
    # Let's build the decomposition step by step.
    
    # For each block-level term t (0..6) and sub-level Strassen term s (0..6):
    # The rank index is r = t * 7 + s
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Get the A-blocks for this term
            a_terms = a_blocks[t]  # list of (block_row, block_col, coeff)
            b_terms = b_blocks[t]  # list of (block_row, block_col, coeff)
            
            # Form the 2x2 matrix X (A-side) and Y (B-side)
            # X[i,j] = sum of coeff * A[block_row, block_col] for terms where (block_row, block_col) maps to (i,j)
            # But actually, the A-blocks are already 2x2 matrices, so we need to
            # form a 2x2 matrix of 2x2 matrices, then flatten.
            
            # Let's think differently. The product we're computing is:
            # (sum of A-blocks with coeffs) * (sum of B-blocks with coeffs)
            # where each "block" is a 2x2 matrix.
            
            # So we have X = sum_t coeff_t * A_t and Y = sum_t coeff_t * B_t
            # where A_t and B_t are 2x2 matrices.
            
            # The product X * Y is a 2x2 matrix of 2x2 matrices.
            # We want to decompose this using Strassen.
            
            # Strassen on 2x2 matrices (where entries are 2x2 matrices):
            # M0 = (X00 + X11) * (Y00 + Y11)
            # M1 = (X10 + X11) * Y00
            # M2 = X00 * (Y01 - Y11)
            # M3 = X11 * (Y10 - Y00)
            # M4 = (X00 + X01) * Y11
            # M5 = (X10 - X00) * (Y00 + Y10)
            # M6 = (X01 - X11) * (Y10 + Y11)
            
            # Each M is a 2x2 matrix of 2x2 matrices, which we then combine
            # to form the final 4x4 result.
            
            # For the tensor decomposition, each rank-1 term corresponds to
            # a specific (i, j, k) pattern in U, V, W.
            
            # Let's define the Strassen coefficients for the sub-level.
            # For term s, we have specific linear combinations of X and Y entries.
            
            # X entries: X00, X01, X10, X11 (each is a 2x2 matrix)
            # Y entries: Y00, Y01, Y10, Y11 (each is a 2x2 matrix)
            
            # For sub-level Strassen term s:
            # We need to express M_s as a product of two 2x2 matrices
            # (which are linear combinations of X and Y entries).
            
            # Let's define the coefficients for X and Y for each Strassen term.
            # X_coeffs[s] = dict mapping (row, col) to coefficient
            # Y_coeffs[s] = dict mapping (row, col) to coefficient
            
            X_coeffs = [
                {(0,0): 1, (1,1): 1},  # S0: X00 + X11
                {(1,0): 1, (1,1): 1},  # S1: X10 + X11
                {(0,0): 1},             # S2: X00
                {(1,1): 1},             # S3: X11
                {(0,0): 1, (0,1): 1},  # S4: X00 + X01
                {(1,0): 1, (0,0): -1}, # S5: X10 - X00
                {(0,1): 1, (1,1): -1}, # S6: X01 - X11
            ]
            
            Y_coeffs = [
                {(0,0): 1, (1,1): 1},  # S0: Y00 + Y11
                {(0,0): 1},             # S1: Y00
                {(0,1): 1, (1,1): -1}, # S2: Y01 - Y11
                {(1,0): 1, (0,0): -1}, # S3: Y10 - Y00
                {(1,1): 1},             # S4: Y11
                {(0,0): 1, (1,0): 1},  # S5: Y00 + Y10
                {(1,0): 1, (1,1): 1},  # S6: Y10 + Y11
            ]
            
            # Now, X[i,j] = sum of coeff * A[block_row, block_col]
            # where (block_row, block_col) corresponds to the (i,j) entry.
            # Actually, X is a 2x2 matrix where each entry is a 2x2 matrix.
            # X[i,j] = sum over a_terms where (block_row, block_col) = (i,j) of coeff * A_block
            
            # Let's build the U, V, W contributions.
            
            # For the A-side (U):
            # We need to form the 2x2 matrix X, where X[i,j] is a 2x2 matrix.
            # X[i,j] = sum of coeff * A[2*i + si, 2*j + sj] for si, sj in {0,1}
            
            # For a given sub-level Strassen term s, the A-side is:
            # X_s = sum_{(i,j) in X_coeffs[s]} X_coeffs[s][(i,j)] * X[i,j]
            # = sum_{(i,j) in X_coeffs[s]} X_coeffs[s][(i,j)] * sum_{a_terms} coeff_a * A[2*i + si, 2*j + sj]
            
            # This is getting complex. Let's just enumerate all the scalar
            # contributions.
            
            # For U: element (i, j) where i in {0..3}, j in {0..3}
            # U[i*m + j, r] = coefficient
            
            # The A-side for rank term r is a linear combination of A's elements.
            # Let's compute the coefficient for each A[i,j].
            
            U_row = np.zeros(n * m)
            V_row = np.zeros(m * p)
            W_row = np.zeros(p * n)
            
            # For the A-side:
            # For each (i,j) in X_coeffs[s]:
            #   For each (block_row, block_col, coeff) in a_terms:
            #     If (block_row, block_col) == (i, j):
            #       For si in {0,1}, sj in {0,1}:
            #         U[2*i + si, 2*j + sj] += X_coeffs[s][(i,j)] * coeff
            
            for (xi, xj), xc in X_coeffs[s].items():
                for (br, bc, ac) in a_terms:
                    if br == xi and bc == xj:
                        for si in range(2):
                            for sj in range(2):
                                U_row[2*xi + si, 2*xj + sj] += xc * ac
            
            # For the B-side:
            # Similarly, for each (i,j) in Y_coeffs[s]:
            #   For each (block_row, block_col, coeff) in b_terms:
            #     If (block_row, block_col) == (i, j):
            #       For si in {0,1}, sj in {0,1}:
            #         V[2*i + si, 2*j + sj] += Y_coeffs[s][(i,j)] * coeff
            
            for (yi, yj), yc in Y_coeffs[s].items():
                for (br, bc, bc_coeff) in b_terms:
                    if br == yi and bc == yj:
                        for si in range(2):
                            for sj in range(2):
                                V_row[2*yi + si, 2*yj + sj] += yc * bc_coeff
            
            # For the W-side:
            # The result C = X * Y is a 2x2 matrix of 2x2 matrices.
            # C[i,j] = M_s[i,j] where M_s is the Strassen term.
            
            # For sub-level Strassen term s, the result is:
            # If s == 0: C00 += M0, C11 += M0
            # If s == 1: C10 += M1, C11 -= M1
            # If s == 2: C01 += M2, C11 += M2
            # If s == 3: C00 += M3, C10 += M3
            # If s == 4: C00 -= M4, C01 += M4
            # If s == 5: C00 += M5, C10 += M5
            # If s == 6: C11 += M6
            
            # But M_s is a 2x2 matrix of 2x2 matrices.
            # M_s[i,j] is a 2x2 matrix.
            
            # Actually, each M_s is the product of two 2x2 matrices (which are
            # linear combinations of X and Y entries). The product is a 2x2
            # matrix where each entry is a 2x2 matrix.
            
            # For the tensor decomposition, we need to express M_s as a sum
            # of rank-1 terms. But M_s is already a product of two 2x2 matrices,
            # so it's a single rank-1 term at the 2x2 level, which expands to
            # 4 rank-1 terms at the scalar level.
            
            # Wait, I think I'm overcomplicating this. Let me reconsider.
            
            # The key insight is that each Strassen term at the sub-level is
            # a product of two 2x2 matrices. Each 2x2 matrix is a linear
            # combination of the original 4x4 matrix's 2x2 sub-blocks.
            
            # So for rank term r = t*7 + s:
            # U_row is a 16-vector (flattened 4x4 matrix) representing the
            # linear combination of A's elements.
            # V_row is a 16-vector representing the linear combination of B's elements.
            # W_row should be such that W[k*n+i, r] gives the coefficient for
            # C[i,k] in the rank-1 term.
            
            # The rank-1 term is: sum_{i,j,k} U[i,j,r] * V[j,k,r] * W[k,i,r] * A[i,j] * B[j,k]
            # = sum_{i,k} (sum_j U[i,j,r] * V[j,k,r]) * W[k,i,r] * C[i,k]
            
            # So W[k,i,r] should be the coefficient for C[i,k].
            
            # For the sub-level Strassen term s, the result contributes to
            # specific C blocks. Let's define the W contributions.
            
            # The Strassen term s produces a 2x2 matrix of 2x2 matrices.
            # Let's call this R_s. R_s[i,j] is a 2x2 matrix.
            
            # The contribution to C is:
            # For s == 0: C[0,0] += R_s[0,0], C[1,1] += R_s[1,1]
            # For s == 1: C[1,0] += R_s[1,0], C[1,1] -= R_s[1,1]
            # For s == 2: C[0,1] += R_s[0,1], C[1,1] += R_s[1,1]
            # For s == 3: C[0,0] += R_s[0,0], C[1,0] += R_s[1,0]
            # For s == 4: C[0,0] -= R_s[0,0], C[0,1] += R_s[0,1]
            # For s == 5: C[0,0] += R_s[0,0], C[1,0] += R_s[1,0]
            # For s == 6: C[1,1] += R_s[1,1]
            
            # But R_s[i,j] is the (i,j) entry of the product of two 2x2 matrices.
            # Each entry is a scalar (since we're at the sub-block level).
            
            # Actually, I realize that R_s is a 2x2 matrix where each entry
            # is a 2x2 matrix of scalars. So R_s[i,j] is a 2x2 matrix.
            
            # Let's define the W contributions more carefully.
            
            # For the block-level term t, the result contributes to C blocks
            # according to the block-level Strassen combination.
            
            # Block-level Strassen result:
            # C00 = M0 + M3 - M4 + M5
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M6
            
            # Where M_t is the result of block-level term t, which is a 2x2
            # matrix of 2x2 matrices.
            
            # So for block-level term t, the contribution to C blocks is:
            # If t == 0: C00 += M0, C11 += M0
            # If t == 1: C10 += M1, C11 -= M1
            # If t == 2: C01 += M2, C11 += M2
            # If t == 3: C00 += M3, C10 += M3
            # If t == 4: C00 -= M4, C01 += M4
            # If t == 5: C00 += M5, C10 += M5
            # If t == 6: C11 += M6
            
            # And M_t[i,j] is a 2x2 matrix, which is the product of two 2x2
            # matrices (from the sub-level Strassen).
            
            # For the sub-level Strassen term s, M_t[i,j] = X_s[i,j] * Y_s[i,j]
            # where X_s and Y_s are the 2x2 matrices from the sub-level Strassen.
            
            # Actually, M_t is the product of two 2x2 matrices, so M_t[i,j]
            # is a scalar (the (i,j) entry of the product).
            
            # Let's just compute the W_row directly.
            
            # For rank term r = t*7 + s:
            # The contribution to C[i,k] is:
            #   sum_{j} U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # We need W[k,i,r] such that this equals the correct coefficient.
            
            # Let's compute it by constructing the full tensors and solving.
            
            # Actually, let's just verify the decomposition numerically and
            # adjust W if needed.
            
            # For now, let's set W_row based on the expected contribution.
            
            # The product of the A-side and B-side should give the correct
            # C block. Let's compute the expected C block.
            
            # A_side is a 4x4 matrix (flattened as U_row)
            # B_side is a 4x4 matrix (flattened as V_row)
            # The product A_side @ B_side should equal W_row reshaped appropriately.
            
            # Let's compute:
            A_side = U_row.reshape(4, 4)
            B_side = V_row.reshape(4, 4)
            C_side = A_side @ B_side
            
            # W_row should be the transpose of C_side, flattened.
            # W[k*n + i, r] = C_side[i, k]
            W_row = C_side.T.flatten()
            
            U[r] = U_row
            V[r] = V_row
            W[r] = W_row
    
    # Convert to lists for the return value
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
