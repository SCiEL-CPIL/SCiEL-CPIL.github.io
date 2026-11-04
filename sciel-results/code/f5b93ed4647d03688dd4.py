import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen on 3x3 blocks
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # a_idx: index in U (i*m+j)
    # b_idx: index in V (j*p+k)
    # c_idx: index in W (k*n+i)
    # coeff_a, coeff_b, coeff_c
    def add_term(a_idx, b_idx, c_idx, coeff_a, coeff_b, coeff_c):
        nonlocal col
        U[a_idx][col] = coeff_a
        V[b_idx][col] = coeff_b
        W[c_idx][col] = coeff_c
        col += 1

    # Strassen terms for 2x2 blocks of 3x3 matrices
    # Blocks:
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # We define the 7 multiplications
    # Each multiplication is of two 3x3 matrices (linear combos)
    # We iterate through the 7 terms
    
    # Term definitions: (Left Matrix Linear Combo, Right Matrix Linear Combo, Contribution to C blocks)
    # Left/Right are 3x3 matrices.
    # We represent a linear combo of a 3x3 block by a list of (block_id, coeff)
    # block_id: 0=A11, 1=A12, 2=A21, 3=A22
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the 3x3 multiplication to scalar terms.
    # For a product L * R, where L and R are 3x3 linear combos:
    # L_{pq} = sum_{bL} cL[bL] * L_block[bL]_{pq}
    # R_{rs} = sum_{bR} cR[bR] * R_block[bR]_{rs}
    # (L*R)_{uv} = sum_k L_{uk} R_{kv}
    
    # We will iterate u, v, k in 0..2
    # For each, we add terms for all combinations of blocks in L and R
    
    def process_multiplication(left_blocks, right_blocks, c_blocks):
        # left_blocks: list of (block_id, coeff)
        # right_blocks: list of (block_id, coeff)
        # c_blocks: list of (block_id, coeff) for the result C block
        
        for u in range(3):
            for v in range(3):
                for k in range(3):
                    # L_{uk} and R_{kv}
                    # L_{uk} is sum of coeffs * A_block[uk]
                    # R_{kv} is sum of coeffs * B_block[kv]
                    
                    # Indices in global A/B
                    # Block 0 (A11): row u, col k
                    # Block 1 (A12): row u, col k+3
                    # Block 2 (A21): row u+3, col k
                    # Block 3 (A22): row u+3, col k+3
                    
                    # Block 0 (B11): row k, col v
                    # Block 1 (B12): row k, col v+3
                    # Block 2 (B21): row k+3, col v
                    # Block 3 (B22): row k+3, col v+3
                    
                    for bL, cL in left_blocks:
                        a_row = u if bL in [0, 1] else u + 3
                        a_col = k if bL in [0, 2] else k + 3
                        a_idx = a_row * 6 + a_col
                        
                        for bR, cR in right_blocks:
                            b_row = k if bR in [0, 1] else k + 3
                            b_col = v if bR in [0, 2] else v + 3
                            b_idx = b_row * 6 + b_col
                            
                            # This term A_{a_idx} * B_{b_idx} contributes to C blocks
                            # Only if a_col == b_row (which is true by construction of matrix mult indices? 
                            # No, a_col is k or k+3. b_row is k or k+3.
                            # They match if (bL in [0,1] and bR in [0,1]) or (bL in [2,3] and bR in [2,3])
                            # If they don't match, this term is a "cross term" that doesn't directly contribute to C_{i,j} 
                            # in the standard sense, BUT it is part of the linear combination that sums to 0 or is handled 
                            # by the fact that we are building the tensor.
                            # Wait, if a_col != b_row, the term A_{i,j} B_{j',k'} with j != j' is not in the tensor T.
                            # So it must cancel out in the total sum.
                            # However, in a rank-1 decomposition, we add U, V, W entries.
                            # If we add a term where a_col != b_row, it contributes to U and V but not to the correct W?
                            # No, W is determined by the C block contribution.
                            # The term is added to the C block specified by c_blocks.
                            # So we add the term to U, V, and W regardless of a_col == b_row.
                            # The correctness relies on the sum of all such terms (including cross terms) equaling T.
                            
                            for bC, cC in c_blocks:
                                # C block index
                                # bC 0: C11 (rows 0-2, cols 0-2) -> c_idx = v*6 + u
                                # bC 1: C12 (rows 0-2, cols 3-5) -> c_idx = (v+3)*6 + u
                                # bC 2: C21 (rows 3-5, cols 0-2) -> c_idx = v*6 + (u+3)
                                # bC 3: C22 (rows 3-5, cols 3-5) -> c_idx = (v+3)*6 + (u+3)
                                
                                if bC == 0:
                                    c_row = v
                                    c_col = u
                                elif bC == 1:
                                    c_row = v + 3
                                    c_col = u
                                elif bC == 2:
                                    c_row = v
                                    c_col = u + 3
                                else:
                                    c_row = v + 3
                                    c_col = u + 3
                                    
                                c_idx = c_row * 6 + c_col
                                
                                add_term(a_idx, b_idx, c_idx, cL, cR, cC)

    # Define the 7 terms
    # B11: block 0, B12: block 1, B21: block 2, B22: block 3
    
    # M1
    process_multiplication([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]) # C11, C22
    
    # M2
    process_multiplication([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]) # C21
    
    # M3
    process_multiplication([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]) # C12
    
    # M4
    process_multiplication([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]) # C11, C21
    
    # M5
    process_multiplication([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]) # C11, C12
    
    # M6
    process_multiplication([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]) # C22
    
    # M7
    process_multiplication([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]) # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
