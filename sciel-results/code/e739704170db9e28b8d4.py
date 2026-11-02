"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We use a block-decomposition strategy.
    # The problem 2x3x8 is treated as two 2x3x4 problems.
    # Each 2x3x4 problem is embedded into a 4x4x4 problem with zeros.
    # The 4x4x4 multiplication is decomposed using Strassen's structure,
    # but the 3/7 terms vanish due to the zero padding of A.
    # The remaining 4 terms are 2x2x2 multiplications, implemented with schoolbook rank 4.
    # Total rank: 2 * (4 * 4) = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 schoolbook multiplication
    # Inputs:
    #   A_idx: list of 4 indices for the 2x2 A matrix (row-major: 0,1,2,3)
    #   B_idx: list of 4 indices for the 2x2 B matrix
    #   C_rows, C_cols: target indices in W
    #   signs: coefficients for A and B terms
    def add_2x2_mult(A_coeffs, B_coeffs, W_coeffs, col_offset):
        # A_coeffs is a list of 4 tuples (idx, coeff)
        # B_coeffs is a list of 4 tuples (idx, coeff)
        # W_coeffs is a list of 4 tuples (idx, coeff)
        # Schoolbook: 4 multiplications
        for a_idx, a_coeff in A_coeffs:
            if a_coeff == 0: continue
            for b_idx, b_coeff in B_coeffs:
                if b_coeff == 0: continue
                # This logic is flawed for tensor decomposition.
                # Tensor decomposition requires linear combinations.
                # We must pre-compute the linear combinations for the 4x4 Strassen step.
                pass

    # We need to explicitly construct the linear combinations.
    # Let's define the indices.
    # A is 2x3. Indices 0..5.
    # Row 0: 0, 1, 2. Row 1: 3, 4, 5.
    # Padded A' 4x4:
    # A11: [[0, 1], [3, 4]]
    # A12: [[2, 6], [5, 7]] -> 6,7 are zeros. So A12 = [[2,0],[5,0]]
    # A21: zeros.
    # A22: zeros.
    
    # B is 3x8. Indices 0..23.
    # Block 1 (cols 0-3):
    # B11: [[0,1],[3,4]] (Rows 0,1; Cols 0,1)
    # B12: [[2,5],[6,7]] (Rows 0,1; Cols 2,3) -> Wait, B is 3x4.
    # B padded 4x4:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15 (Zeros)
    # B11: [[0,1],[4,5]]
    # B12: [[2,3],[6,7]]
    # B21: [[8,9],[12,13]] -> 12,13 are zeros. So B21 = [[8,9],[0,0]]
    # B22: [[10,11],[14,15]] -> 14,15 are zeros. So B22 = [[10,11],[0,0]]
    
    # Block 2 (cols 4-7):
    # B'2 Row 0: 16,17,18,19
    # B'2 Row 1: 20,21,22,23
    # B'2 Row 2: 24,25,26,27
    # B'2 Row 3: Zeros
    # B21: [[16,17],[20,21]] -> Wait, indices map to B matrix.
    # B is 3x8.
    # Block 2 corresponds to B[0:3, 4:8].
    # Padded B2' 4x4:
    # Row 0: 16,17,18,19 (from B[0, 4..7])
    # Row 1: 20,21,22,23 (from B[1, 4..7])
    # Row 2: 24,25,26,27 (from B[2, 4..7])
    # Row 3: Zeros
    
    # We have 4 products for each block.
    # Product 1: A11 * (B11 + B22)
    # Product 3: A11 * (B12 - B22)
    # Product 5: (A11 + A12) * B22
    # Product 7: (A11 - A12) * (B11 + B21)
    
    # Let's implement this by generating the U, V, W entries.
    # We will create a list of "terms" (a, b, c, coeff_a, coeff_b, coeff_c)
    # where term is a * b -> c.
    # Then we expand linear combinations.
    
    # To simplify, we will just hardcode the 32 multiplications.
    # This is safer and faster.
    
    col = 0
    def add_mult(a_idx, b_idx, c_idx, coeff=1):
        nonlocal col
        U[a_idx][col] = coeff
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1

    # We need to handle linear combinations in U and V.
    # E.g., A11 + A12.
    # U for (A11+A12) will have non-zeros at indices 0,1,3,4,2,5.
    
    # Let's define the index mappings.
    # A indices:
    # A11: 0,1,3,4
    # A12: 2,5 (and 6,7 are 0)
    
    # B Block 1 indices:
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9 (12,13 are 0)
    # B22: 10,11 (14,15 are 0)
    
    # B Block 2 indices:
    # B11: 16,17,20,21
    # B12: 18,19,22,23
    # B21: 24,25 (28,29 are 0)
    # B22: 26,27 (30,31 are 0)
    
    # W indices (2x8):
    # W[k*n + i]. n=2.
    # Block 1 W (cols 0-3):
    # C11 (top-left 2x2): 0, 1, 2, 3?
    # C is 2x4.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3?
    # W index = k*n + i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # So C11 corresponds to W indices 0, 1, 2, 3.
    # C12 (top-right 2x2):
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    # So C12 corresponds to W indices 4, 5, 6, 7.
    
    # Block 2 W (cols 4-7):
    # C21 (top-left of block 2, which is global cols 4,5? No, block 2 is cols 4-7.
    # The 2x2 blocks of the 2x4 result C2:
    # C21 (global cols 4,5):
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    # k=5, i=0 -> 10
    # k=5, i=1 -> 11
    # C22 (global cols 6,7):
    # k=6, i=0 -> 12
    # k=6, i=1 -> 13
    # k=7, i=0 -> 14
    # k=7, i=1 -> 15
    
    # We need to map the Strassen outputs to these W indices.
    # M = A'B'
    # M11 = P1 + P4 - P5 + P7
    # M12 = P3 + P5
    # M21 = P2 + P4
    # M22 = P1 - P2 + P3 + P6
    
    # In our case:
    # P2, P4, P6 are 0.
    # M11 = P1 - P5 + P7
    # M12 = P3 + P5
    # M21 = 0
    # M22 = P1 + P3
    
    # Wait, M21 corresponds to the bottom-left 2x2 of the 4x4 result.
    # But our original result C is 2x8.
    # The top-left 2x2 of M (M11) corresponds to C[0:2, 0:2].
    # The top-right 2x2 of M (M12) corresponds to C[0:2, 2:4].
    # The bottom-left 2x2 (M21) corresponds to C[2:4, 0:2] which are zeros in our 2x8 result?
    # No, the result is 2x8. The "bottom" rows of the 4x4 result are not part of the original output.
    # So we only care about M11 and M12.
    
    # So for each block, we need to compute:
    # 1. M11 = P1 - P5 + P7
    # 2. M12 = P3 + P5
    
    # P1 = A11 * (B11 + B22)
    # P3 = A11 * (B12 - B22)
    # P5 = (A11 + A12) * B22
    # P7 = (A11 - A12) * (B11 + B21)
    
    # We will implement these 4 products. Each is a 2x2x2 multiplication.
    # We use schoolbook rank 4 for each.
    # Total 16 per block. 32 total.
    
    # To avoid complex linear combination expansion, we will just add the 4 scalar multiplications for each product.
    # BUT, the inputs to the scalar multiplications are linear combinations.
    # E.g., P1 involves (B11+B22).
    # We cannot just add scalar multiplications directly unless we expand the linear combos.
    
    # Let's expand P1 = A11 * (B11 + B22).
    # A11 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
    # Product is sum of 8 terms? No, rank 4 means 4 terms.
    # Schoolbook 2x2:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # This requires 4 multiplications.
    # The "b" matrix is B11+B22.
    # So b00 = B11[0,0] + B22[0,0].
    # This means in the V matrix, for this multiplication, we have non-zeros at indices for B11[0,0] and B22[0,0].
    
    # We will write a helper to add a schoolbook 2x2 multiplication with linear combo inputs.
    
    def add_2x2_schoolbook(A_indices, B_indices, C_indices, col_start):
        # A_indices: 4 indices for A (0,0; 0,1; 1,0; 1,1)
        # B_indices: 4 indices for B (0,0; 0,1; 1,0; 1,1)
        # C_indices: 4 indices for C (0,0; 0,1; 1,0; 1,1)
        # Returns new col offset
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # This standard schoolbook doesn't allow linear combos in A or B directly in one column.
        # We must distribute.
        # If B is a linear combo, say B = X + Y, then A*B = A*X + A*Y.
        # This doubles the rank.
        
        # So, if B11+B22 is the input, we must do A11*B11 + A11*B22.
        # That's 2 * 4 = 8 multiplications.
        # This ruins the rank benefit!
        
        # We need a decomposition of the 2x2x2 tensor that allows linear combinations?
        # No, the tensor rank is the minimum number of rank-1 tensors.
        # If we replace B with B1+B2, we are computing A*(B1+B2).
        # This is equivalent to adding two rank-4 tensors.
        # So the rank is additive.
        
        # Therefore, P1 = A11*(B11+B22) has rank rank(A11*B11) + rank(A11*B22) = 4 + 4 = 8.
        # P3 = A11*(B12-B22) has rank 8.
        # P5 = (A11+A12)*B22 has rank 8.
        # P7 = (A11-A12)*(B11+B21) has rank 8 + 8 = 16?
        # A11*B11 + A11*B21 - A12*B11 - A12*B21.
        # 4 terms, each rank 4. Total 16.
        
        # Total rank per block: 8 + 8 + 8 + 16 = 40.
        # Two blocks: 80.
        # Worse than 48.
        
        return col_start

    # This approach is flawed due to linearity distribution.
    # We must find a decomposition that minimizes the rank considering the specific structure.
    
    # Fallback: Use the baseline schoolbook (rank 48) to ensure validity, 
    # but this doesn't improve the score.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
