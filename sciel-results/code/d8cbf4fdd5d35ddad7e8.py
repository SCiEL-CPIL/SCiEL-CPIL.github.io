import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # We exploit the dimension mismatch by partitioning the 3x5x6 tensor.
    # The directive suggests: 3x5x6 = 2x4x5 block + 1x1x1 block.
    # Rank(2x4x5) = 17, Rank(1x1x1) = 1. Total Rank = 18.
    
    # We need to explicitly construct the rank-17 decomposition for 2x4x5.
    # A 2x4 matrix A and a 4x5 matrix B produce a 2x5 matrix C.
    # Let A be 2x4, B be 4x5.
    # We can view 2x4x5 as a sub-tensor of 2x4x4 (rank 14) + 2x4x1 (rank 4)?
    # R(2,4,4) <= 2 * R(2,2,4)? No.
    # R(2,4,4) <= R(2,2,4) + R(2,2,4)? No.
    # Let's use the known bound R(2,4,5) <= 17.
    # Actually, R(2,4,5) is known to be 17.
    # How to construct it?
    # R(2,4,5) <= R(2,4,4) + R(2,4,1) = 14 + 4 = 18? No, R(2,4,4)=14, R(2,4,1)=4. 14+4=18.
    # But the claim is 17.
    # R(2,4,5) <= R(2,2,5) + R(2,2,5) ?
    # R(2,2,5) <= R(2,2,4) + R(2,2,1) = 7 + 2 = 9?
    # Then 2*9 = 18.
    # 
    # Let's look for a specific rank-17 decomposition.
    # It is known that R(2,4,5) = 17.
    # One construction:
    # Split the 4 columns of A into two 2-column blocks: A = [A1, A2]
    # Split the 4 rows of B into two 2-row blocks: B = [B1; B2]
    # C = A1 B1 + A2 B2.
    # This is a 2x2x5 multiplication? No.
    # A1 is 2x2, B1 is 2x5. C1 = A1 B1 is 2x5.
    # A2 is 2x2, B2 is 2x5. C2 = A2 B2 is 2x5.
    # C = C1 + C2.
    # R(2,2,5) <= R(2,2,4) + R(2,2,1) = 7 + 2 = 9.
    # So R(2,4,5) <= 9 + 9 = 18.
    # 
    # To get 17, we need a better split or a specific identity.
    # R(2,4,5) <= 17 is a known result (e.g. by Landsberg).
    # However, constructing the explicit coefficients for a rank-17 decomposition is non-trivial.
    # 
    # Alternative:
    # Use the schoolbook for the 1x1x1 part (rank 1).
    # For the 2x4x5 part, if I can't get 17, I can use 18.
    # Total rank 19.
    # 19 < 90 (schoolbook).
    # Score = 90/19 = 4.73.
    # 
    # Let's try to implement the rank-18 decomposition for 2x4x5 (2 * R(2,2,5)).
    # R(2,2,5) decomposition:
    # We need to decompose 2x2x5.
    # Split 5 into 4+1.
    # R(2,2,5) = R(2,2,4) + R(2,2,1) = 7 + 2 = 9.
    # 
    # R(2,2,4) decomposition:
    # Split 4 into 2+2.
    # R(2,2,4) = R(2,2,2) + R(2,2,2) = 7 + 7 = 14? No.
    # R(2,2,4) <= 2 * R(2,1,4)? No.
    # R(2,2,4) is known to be 9? Or 10?
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14.
    # Actually R(2,2,4) = 9.
    # 
    # Let's stick to a guaranteed valid decomposition, even if not minimal.
    # 
    # Strategy:
    # 1. Decompose 3x5x6 into blocks.
    # 3 = 2 + 1
    # 5 = 4 + 1
    # 6 = 4 + 2
    # 
    # Blocks:
    # (2,4,4), (2,4,2), (2,1,4), (2,1,2), (1,4,4), (1,4,2), (1,1,4), (1,1,2)
    # 
    # R(2,4,4) <= 2 * R(2,2,4) = 2 * 9 = 18? Or R(2,4,4) <= 14.
    # Let's use R(2,4,4) = 14.
    # R(2,4,2) <= 2 * R(2,2,2) = 14? Or R(2,4,2) <= 8.
    # R(2,1,4) <= 4.
    # R(2,1,2) <= 2.
    # R(1,4,4) <= 4.
    # R(1,4,2) <= 2.
    # R(1,1,4) <= 1.
    # R(1,1,2) <= 1.
    # 
    # Total: 14 + 8 + 4 + 2 + 4 + 2 + 1 + 1 = 36.
    # 36 < 90.
    # 
    # Let's implement this block decomposition.
    # We need explicit decompositions for:
    # R(2,4,4) = 14
    # R(2,4,2) = 8
    # R(2,1,4) = 4
    # R(2,1,2) = 2
    # R(1,4,4) = 4
    # R(1,4,2) = 2
    # R(1,1,4) = 1
    # R(1,1,2) = 1
    # 
    # R(2,4,4) = 14:
    # Split A (2x4) into A1, A2 (2x2).
    # Split B (4x4) into B1, B2 (4x2)? No.
    # B is 4x4.
    # C = A1 B1 + A2 B2 where B1, B2 are 4x2?
    # A1 is 2x2, B1 is 2x2?
    # If we split A into two 2x2 blocks horizontally: A = [A1, A2].
    # Split B vertically: B = [B1; B2] where B1, B2 are 2x4.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x4. A2 B2 is 2x4.
    # R(2,2,4) = 9.
    # So R(2,4,4) <= 9 + 9 = 18.
    # But R(2,4,4) = 14 is known.
    # 
    # Let's use the 14-term decomposition for 2x4x4.
    # It is derived from R(2,2,4)=9? No, 2*9=18.
    # R(2,4,4) = 14 is a specific result.
    # 
    # To be safe and fast, I will implement a decomposition using Strassen for 2x2x2 blocks.
    # 
    # Let's use a simpler, valid decomposition:
    # 3x5x6.
    # Split 3=2+1, 5=3+2, 6=3+3.
    # 
    # Actually, let's just use the schoolbook but with a smaller effective rank by exploiting zeros?
    # No, the tensor is dense.
    # 
    # Let's implement the rank-36 decomposition above, but I need the coefficients.
    # 
    # R(2,2,4) = 9 decomposition:
    # A (2x2), B (2x4), C (2x4).
    # Split B into B1 (2x2), B2 (2x2).
    # C1 = A B1, C2 = A B2.
    # R(2,2,2) = 7.
    # R(2,2,4) <= 7 + 7 = 14.
    # But R(2,2,4) = 9.
    # 
    # I will use the rank-14 decomposition for 2x2x4 (2 Strassen blocks).
    # Then R(2,4,4) <= 14 + 14 = 28.
    # 
    # This is getting complicated. Let's go with a known valid decomposition:
    # R(3,5,6) <= R(3,5,5) + R(3,5,1) = R(3,5,5) + 3.
    # R(3,5,5) <= R(3,4,5) + R(3,1,5) = R(3,4,5) + 3.
    # R(3,4,5) <= R(3,4,4) + R(3,4,1) = R(3,4,4) + 3.
    # R(3,4,4) <= R(3,2,4) + R(3,2,4) = 2 * R(3,2,4).
    # R(3,2,4) <= R(3,2,2) + R(3,2,2) = 2 * R(3,2,2).
    # R(3,2,2) <= R(3,1,2) + R(3,1,2) = 2 * R(3,1,2) = 2 * 2 = 4?
    # R(3,1,2) = 2.
    # So R(3,2,2) = 4.
    # R(3,2,4) = 8.
    # R(3,4,4) = 16.
    # R(3,4,5) = 19.
    # R(3,5,5) = 22.
    # R(3,5,6) = 25.
    # 
    # Let's verify R(3,1,2) = 2.
    # A (3x1), B (1x2), C (3x2).
    # C_i1 = A_i B_11, C_i2 = A_i B_12.
    # Terms: A_i B_11, A_i B_12.
    # Rank 2. Correct.
    # 
    # R(3,2,2) = 4.
    # A (3x2), B (2x2), C (3x2).
    # C = A B.
    # C_i1 = A_i1 B_11 + A_i2 B_21.
    # C_i2 = A_i1 B_12 + A_i2 B_22.
    # We can use Strassen-like?
    # R(3,2,2) <= R(3,2,1) + R(3,2,1) = 2 * 2 = 4.
    # R(3,2,1) = 2.
    # So R(3,2,2) = 4.
    # 
    # So the rank is 25.
    # 
    # Let's implement this decomposition.
    # 
    # We need to map the indices.
    # 
    # Function to add a block decomposition.
    # 
    # Block (n_block, m_block, p_block) with rank r_block.
    # We need to specify the U, V, W entries for this block.
    # 
    # For R(3,1,2) = 2:
    # A is 3x1. B is 1x2. C is 3x2.
    # U: 3x1 -> 3 entries.
    # V: 1x2 -> 2 entries.
    # W: 2x3 -> 6 entries.
    # 
    # Term 1: A_i * B_11 -> C_i1
    # U[i, 0] = 1
    # V[0, 0] = 1
    # W[0, i] = 1 (for C_i1, index is 0*3 + i = i)
    # 
    # Term 2: A_i * B_12 -> C_i2
    # U[i, 1] = 1
    # V[0, 1] = 1
    # W[1, i] = 1 (for C_i2, index is 1*3 + i = 3 + i)
    # 
    # For R(3,2,2) = 4:
    # A is 3x2. B is 2x2. C is 3x2.
    # Split B into B_col1 (2x1) and B_col2 (2x1).
    # C_col1 = A B_col1. C_col2 = A B_col2.
    # R(3,2,1) = 2.
    # Term 1: A_i1 * B_11 -> C_i1
    # Term 2: A_i2 * B_21 -> C_i1
    # Term 3: A_i1 * B_12 -> C_i2
    # Term 4: A_i2 * B_22 -> C_i2
    # 
    # For R(3,2,4) = 8:
    # Split B (2x4) into B1 (2x2) and B2 (2x2).
    # C1 = A B1. C2 = A B2.
    # R(3,2,2) = 4.
    # So 8 terms.
    # 
    # For R(3,4,4) = 16:
    # Split A (3x4) into A1 (3x2) and A2 (3x2).
    # Split B (4x4) into B1 (2x4) and B2 (2x4).
    # C1 = A1 B1. C2 = A2 B2.
    # R(3,2,4) = 8.
    # So 16 terms.
    # 
    # For R(3,4,5) = 19:
    # Split B (4x5) into B1 (4x4) and B2 (4x1).
    # C1 = A B1. C2 = A B2.
    # R(3,4,4) = 16.
    # R(3,4,1) = 3.
    # Total 19.
    # 
    # For R(3,5,5) = 22:
    # Split A (3x5) into A1 (3x4) and A2 (3x1).
    # Split B (5x5) into B1 (4x5) and B2 (1x5).
    # C1 = A1 B1. C2 = A2 B2.
    # R(3,4,5) = 19.
    # R(3,1,5) = 3.
    # Total 22.
    # 
    # For R(3,5,6) = 25:
    # Split B (5x6) into B1 (5x5) and B2 (5x1).
    # C1 = A B1. C2 = A B2.
    # R(3,5,5) = 22.
    # R(3,5,1) = 3.
    # Total 25.
    # 
    # This is a valid decomposition of rank 25.
    # 
    # Implementation:
    # We will build U, V, W by adding terms for each sub-block.
    # 
    # We need to map the global indices to local indices.
    # 
    # Global A: 3x5. Global B: 5x6. Global C: 3x6.
    # 
    # Step 1: R(3,5,6)
    # B1: rows 0-4, cols 0-4. (5x5)
    # B2: rows 0-4, cols 5.   (5x1)
    # 
    # Step 2: R(3,5,5)
    # A1: rows 0-2, cols 0-3. (3x4)
    # A2: rows 0-2, cols 4.   (3x1)
    # B1: rows 0-4, cols 0-4. (5x5)
    # B2: rows 4, cols 0-4.   (1x5)
    # 
    # Wait, the split for R(3,5,5) is A = [A1, A2], B = [B1; B2].
    # A1 is 3x4, A2 is 3x1.
    # B1 is 4x5, B2 is 1x5.
    # 
    # Step 3: R(3,4,5)
    # A1: 3x4.
    # B1: 4x5.
    # Split B1 into B11 (4x4) and B12 (4x1).
    # 
    # Step 4: R(3,4,4)
    # A1: 3x4.
    # B11: 4x4.
    # Split A1 into A11 (3x2) and A12 (3x2).
    # Split B11 into B111 (2x4) and B112 (2x4).
    # 
    # Step 5: R(3,2,4)
    # A11: 3x2.
    # B111: 2x4.
    # Split B111 into B1111 (2x2) and B1112 (2x2).
    # 
    # Step 6: R(3,2,2)
    # A11: 3x2.
    # B1111: 2x2.
    # 
    # This recursive structure is complex to map directly.
    # 
    # Instead, I will implement a generic function that takes a decomposition of a block
    # and adds it to the global matrices with the correct offsets.
    # 
    # But the decompositions are recursive.
    # 
    # Let's just hardcode the terms for the leaf blocks.
    # 
    # Leaf 1: R(3,1,2) = 2.
    # Leaf 2: R(3,2,2) = 4.
    # 
    # Actually, R(3,2,2) is built from two R(3,2,1).
    # R(3,2,1) = 2.
    # 
    # Let's define the base case R(3,1,1) = 1.
    # R(3,1,2) = 2.
    # R(3,2,1) = 2.
    # R(3,2,2) = 4.
    # 
    # I will write a helper to add a term.
    # 
    # U[i*m + j][col] = coeff
    # V[j*p + k][col] = coeff
    # W[k*n + i][col] = coeff
    # 
    # For a block at offset (i0, j0, k0) with size (ni, mj, pk):
    # A index: i0 + i, j0 + j
    # B index: j0 + j, k0 + k
    # C index: i0 + i, k0 + k
    # 
    # U index: (i0+i)*m + (j0+j)
    # V index: (j0+j)*p + (k0+k)
    # W index: (k0+k)*n + (i0+i)
    # 
    # Let's implement the rank-25 decomposition.
    # 
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a single term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1
    
    # R(3,5,6)
    # C = A B1 + A B2
    # B1 is 5x5 (cols 0-4), B2 is 5x1 (col 5)
    
    # Part 2: A B2 (R(3,5,1) = 3)
    # A is 3x5, B2 is 5x1.
    # C_i5 = sum_j A_ij B_j5.
    # Terms: A_ij * B_j5 for j=0..4? No, R(3,5,1) = 3?
    # R(3,5,1) <= 3.
    # C_i5 = A_i0 B_05 + A_i1 B_15 + A_i2 B_25 + A_i3 B_35 + A_i4 B_45.
    # This is rank 5 if done naively.
    # But R(3,5,1) = 3?
    # A (3x5), B (5x1).
    # C (3x1).
    # C_i = sum A_ij B_j.
    # This is an inner product.
    # Rank is 3?
    # C_0 = A_00 B_0 + A_01 B_1 + A_02 B_2 + A_03 B_3 + A_04 B_4
    # C_1 = A_10 B_0 + A_11 B_1 + A_12 B_2 + A_13 B_3 + A_14 B_4
    # C_2 = A_20 B_0 + A_21 B_1 + A_22 B_2 + A_23 B_3 + A_24 B_4
    # 
    # We can write this as:
    # T1 = B_0 (A_00 C_0 + A_10 C_1 + A_20 C_2)? No.
    # 
    # R(3,5,1) = 3 is not correct. R(3,5,1) = 5?
    # No, R(n, m, 1) = n * m? No.
    # R(n, m, 1) = n?
    # C_i = sum_j A_ij B_j.
    # We can group by B_j.
    # T_j = B_j * (A_0j C_0 + A_1j C_1 + A_2j C_2)? No.
    # 
    # Actually, R(3,5,1) = 5.
    # Because we need 5 multiplications B_j * (linear combo of A).
    # 
    # So my rank estimate was wrong.
    # R(3,5,1) = 5.
    # R(3,5,6) <= R(3,5,5) + 5.
    # R(3,5,5) <= R(3,4,5) + R(3,1,5).
    # R(3,1,5) = 3?
    # A (3x1), B (1x5).
    # C (3x5).
    # C_ij = A_i B_j.
    # Rank 3.
    # 
    # So R(3,5,5) <= R(3,4,5) + 3.
    # R(3,4,5) <= R(3,4,4) + R(3,4,1).
    # R(3,4,1) = 4.
    # R(3,4,4) <= 2 * R(3,2,4).
    # R(3,2,4) <= 2 * R(3,2,2).
    # R(3,2,2) = 4.
    # R(3,2,4) = 8.
    # R(3,4,4) = 16.
    # R(3,4,5) = 20.
    # R(3,5,5) = 23.
    # R(3,5,6) = 28.
    # 
    # 28 < 90.
    # 
    # Let's implement rank 28.
    # 
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1
    
    # R(3,5,6)
    # C = A B1 + A B2
    # B1: 5x5 (cols 0-4), B2: 5x1 (col 5)
    
    # Part 2: A B2 (R(3,5,1) = 5)
    # C_i5 = sum_j A_ij B_j5
    for j in range(5):
        for i in range(3):
            add_term(i, j, 5, 1, 1, 1)
            # Wait, C_i5 = A_ij B_j5.
            # U[i,j] = 1, V[j,5] = 1, W[5,i] = 1.
            # This is 15 terms.
            # But R(3,5,1) = 5?
            # No, R(3,5,1) = 5 is for the multiplication of the vectors.
            # C_i = sum_j A_ij B_j.
            # We can define T_j = B_j.
            # Then C_i = sum_j A_ij T_j.
            # This requires 5 multiplications T_j * A_ij? No.
            # The tensor is T_{ijk} = delta_{jk} delta_{ik}? No.
            # 
            # The decomposition is:
            # C_i = sum_j A_ij B_j.
            # Term j: A_ij * B_j.
            # For fixed j, we have terms for all i.
            # U[i,j] = 1, V[j,j] = 1? No, V is 5x1.
            # V[j,0] = 1.
            # W[0,i] = 1? No, W is 1x3.
            # 
            # So for each j:
            # U[i,j] = 1
            # V[j,0] = 1
            # W[0,i] = 1
            # 
            # This is 3 terms per j. Total 15 terms.
            # 
            # So R(3,5,1) = 15?
            # No, we can share the B_j multiplication.
            # T_j = B_j.
            # C_i = sum_j A_ij T_j.
            # This is not a rank-1 term.
            # 
            # A rank-1 term is U_i * V_j * W_k.
            # 
            # For C_i = sum_j A_ij B_j:
            # We can write C_i = A_i0 B_0 + A_i1 B_1 + ...
            # Each A_ij B_j is a rank-1 term?
            # U_ij = 1, V_j = 1, W_i = 1.
            # Yes.
            # So R(3,5,1) = 15.
            
    # This is getting too high.
    # 
    # Let's go back to the schoolbook 90.
    # It is the safest.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
