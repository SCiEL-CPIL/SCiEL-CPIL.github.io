import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Strassen 2x2x2
    # U, V, W for 2x2x2
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen matrices
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # We define the linear combinations for U, V, W
    # U maps A (4x1) to 7 intermediates
    # V maps B (4x1) to 7 intermediates
    # W maps 7 intermediates to C (4x1)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # Col 0: M1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[1, 0] = 1; W_2[2, 0] = 1; W_2[3, 0] = 1
    
    # Col 1: M2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = 1
    
    # Col 2: M3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # Col 3: M4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[2, 3] = 1; W_2[0, 3] = 1
    
    # Col 4: M5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[1, 4] = 1; W_2[3, 4] = 1
    
    # Col 5: M6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[2, 5] = 1
    W_2[0, 5] = 1; W_2[2, 5] = 1
    
    # Col 6: M7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[1, 6] = 1; V_2[3, 6] = 1
    W_2[1, 6] = 1; W_2[0, 6] = 1

    # Step 2: Build 4x4x4 using 2x2x2 blocks
    # A is 4x4, B is 4x4
    # Split into 2x2 blocks
    # U_4 will be 16x49
    # V_4 will be 16x49
    # W_4 will be 16x49
    
    def build_strassen_4x4x4(U_2, V_2, W_2):
        U_4 = np.zeros((16, 49))
        V_4 = np.zeros((16, 49))
        W_4 = np.zeros((16, 49))
        
        # 7 Strassen steps
        # Each step involves 2x2x2 multiplication
        # The 2x2 blocks of A are mapped to 4x1 vectors
        # The 2x2 blocks of B are mapped to 4x1 vectors
        # The result is 7 2x2 matrices, which are combined into 4x4 C
        
        # Mapping of 4x4 index (i,j) to 2x2 block index and local index
        # i = 2*bi + li, j = 2*bj + lj
        # A index = i*4 + j
        # Local A index in 2x2 block = li*2 + lj
        
        # We have 7 global steps.
        # For each global step s (0..6), we have a 2x2x2 decomposition.
        # The inputs to this 2x2x2 decomposition are linear combinations of the 4 A-blocks and 4 B-blocks.
        # This is equivalent to applying U_2, V_2 to the vector of A-blocks and B-blocks.
        
        # Let's construct the full matrices by substitution.
        # A_vec (16x1) is flattened A.
        # We want U_4 such that U_4 @ A_vec gives the 49 intermediate values.
        # The 49 values are 7 groups of 7 (from the 2x2x2 steps).
        # Actually, each of the 7 "M" values is a 2x2 matrix (4 values).
        # So 7 * 4 = 28 values? No.
        # Strassen 4x4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # So total rank is 7*7=49.
        # The 49 intermediates are the 7 scalars from each of the 7 2x2 multiplications.
        
        # So, for each global step s (0..6):
        # We compute M_s = A_s * B_s where A_s, B_s are 2x2 matrices.
        # A_s = sum_{i,j} U_2[local_i, s] * A_block(i,j) ? No.
        # U_2 maps the 4 entries of a 2x2 matrix to 7 intermediates.
        # Here, the "2x2 matrix" A_s is formed by linear combinations of the 4 blocks of A.
        
        # Let's define the block indices for A (4 blocks: 00, 01, 10, 11)
        # A_block 00: indices 0,1,4,5
        # A_block 01: indices 2,3,6,7
        # A_block 10: indices 8,9,12,13
        # A_block 11: indices 10,11,14,15
        
        # For each global step s, we need to form A_s (2x2) and B_s (2x2).
        # The coefficients for A_s come from the s-th column of U_2?
        # No, U_2 is for a single 2x2 matrix.
        # In Strassen, the "A" input to the 2x2x2 subroutine is a 2x2 matrix of blocks.
        # So A_s is not a single 2x2 matrix, but the 2x2x2 subroutine operates on 2x2 matrices.
        # Wait, the recursion is:
        # T_4 = 7 * T_2.
        # The inputs to the 7 T_2 instances are linear combinations of the 16 elements of A and B.
        
        # Let's use the explicit formulas for 4x4 Strassen.
        # It's easier to just apply the 2x2 Strassen logic to the blocks.
        
        # For each of the 7 steps:
        # Step 0: M0 = (A00+A11)*(B00+B11)
        # A0 is (A00+A11), B0 is (B00+B11).
        # A0 is a 2x2 matrix. B0 is a 2x2 matrix.
        # We multiply A0 and B0 using Strassen 2x2.
        # This requires 7 scalar multiplications.
        # The inputs to these 7 scalar multiplications are linear combos of the 4 elements of A0 and 4 elements of B0.
        # The elements of A0 are linear combos of the 4 blocks of A.
        
        # So we can build U_4 by composing the linear maps.
        
        # Map 1: A (16x1) -> 4 blocks (4x4 matrix, flattened 16x1? No, 4 blocks of 4 elements each).
        # Actually, let's just build the 16x49 matrix directly.
        
        # For each global step g (0..6):
        # Define the linear combination of A blocks to form A_g (2x2).
        # Define the linear combination of B blocks to form B_g (2x2).
        # Then apply U_2, V_2, W_2 to A_g, B_g.
        
        # A_g (2x2) elements are sums of A blocks.
        # A_block (i,j) has 4 elements.
        # So A_g (i,j) is a 2x2 matrix? No, A_g is a 2x2 matrix of scalars?
        # No, in the recursion, A_g is a 2x2 matrix.
        # The elements of A_g are linear combinations of the elements of A.
        
        # Let's define the coefficients for A_g and B_g for each global step.
        
        steps = [
            # (coeff_A, coeff_B)
            # coeff is a 4x4 matrix of coefficients for the 4 blocks?
            # No, the blocks are 2x2 matrices.
            # Let's represent the 4 blocks of A as A00, A01, A10, A11.
            # A_g = c00*A00 + c01*A01 + c10*A10 + c11*A11.
            # This results in a 2x2 matrix A_g.
            
            # Step 0: A_g = A00 + A11. B_g = B00 + B11.
            # Step 1: A_g = A10 + A11. B_g = B00.
            # Step 2: A_g = A00. B_g = B01 - B11.
            # Step 3: A_g = A11. B_g = B10 - B00.
            # Step 4: A_g = A00 + A01. B_g = B11.
            # Step 5: A_g = A10 - A00. B_g = B00 + B10.
            # Step 6: A_g = A01 - A11. B_g = B01 + B11.
        ]
        
        # Coefficients for A blocks (indices 0:00, 1:01, 2:10, 3:11)
        A_coeffs = [
            [1, 0, 0, 1], # 00+11
            [0, 0, 1, 1], # 10+11
            [1, 0, 0, 0], # 00
            [0, 0, 0, 1], # 11
            [1, 1, 0, 0], # 00+01
            [1, 0, -1, 0], # 00-10? No, A10-A00 -> -1*00 + 1*10.
            [0, 1, 0, -1]  # 01-11
        ]
        
        B_coeffs = [
            [1, 0, 0, 1], # 00+11
            [1, 0, 0, 0], # 00
            [0, 1, 0, -1], # 01-11
            [0, 0, 1, -1], # 10-00 -> -1*00 + 1*10.
            [0, 0, 0, 1], # 11
            [1, 0, 1, 0], # 00+10
            [0, 1, 0, 1]  # 01+11
        ]
        
        # For each global step g:
        for g in range(7):
            # A_g is formed by combining A blocks.
            # A_block b has 4 elements.
            # A_g (2x2) element (i,j) is sum_b coeff[b] * A_block(b)_{i,j}.
            
            # We need to map the 16 elements of A to the 4 elements of A_g.
            # Then apply U_2 to get 7 intermediates for this step.
            
            # U_2 maps 4 elements (of a 2x2 matrix) to 7 intermediates.
            # Here, the "4 elements" are the 4 elements of A_g.
            # Each element of A_g is a linear combination of 4 blocks.
            
            # Let's construct the 16x7 matrix U_g that maps A (16x1) to the 7 intermediates of step g.
            U_g = np.zeros((16, 7))
            V_g = np.zeros((16, 7))
            W_g = np.zeros((16, 7))
            
            # For each local index l (0..3) in the 2x2 matrix A_g:
            # A_g[l] = sum_{b=0}^3 A_coeffs[g][b] * A_block(b)[l]
            # A_block(b) has 4 elements.
            # A_block(0) (00): indices 0,1,4,5. Local index l corresponds to global index?
            # Block 00: i in 0,1; j in 0,1. Global idx = i*4+j.
            # Local idx l = li*2+lj.
            # Global idx = (2*bi + li)*4 + (2*bj + lj) = 8*bi + 4*li + 2*bj + lj.
            
            # For A:
            for l in range(4):
                li = l // 2
                lj = l % 2
                for b in range(4):
                    bi = b // 2
                    bj = b % 2
                    global_idx = 8*bi + 4*li + 2*bj + lj
                    # A_g[l] += coeff * A[global_idx]
                    # U_2[l, s] is coeff for A_g[l] in intermediate s.
                    for s in range(7):
                        U_g[global_idx, s] += U_2[l, s] * A_coeffs[g][b]
                        
            # Similarly for V_g
            for l in range(4):
                li = l // 2
                lj = l % 2
                for b in range(4):
                    bi = b // 2
                    bj = b % 2
                    global_idx = 8*bi + 4*li + 2*bj + lj
                    for s in range(7):
                        V_g[global_idx, s] += V_2[l, s] * B_coeffs[g][b]
                        
            # For W_g:
            # W_2 maps 7 intermediates to 4 elements of C_g (2x2).
            # C_g is added to C.
            # C_g (2x2) elements are placed in C at blocks determined by g?
            # No, C is the sum of 7 2x2 matrices M_g.
            # So C = M_0 + ... + M_6.
            # Each M_g is a 2x2 matrix.
            # So W_g should add the contribution of M_g to the corresponding blocks of C.
            # Since C is the sum, the position in C is the same for all g.
            # So W_g maps the 7 intermediates to the 4 elements of C_g, and these are added to C.
            # So W_g is just W_2, but expanded to 16x7?
            # No, W_2 is 4x7.
            # The 4 elements of C_g correspond to the 4 blocks of C?
            # No, M_g is a 2x2 matrix.
            # In the 4x4 result C, M_g is added to the whole C?
            # No, Strassen 4x4 produces C = M_0 + ... + M_6.
            # Each M_g is a 2x2 matrix? No, M_g is a 4x4 matrix?
            # No, in Strassen 2x2, M is a 2x2 matrix.
            # In Strassen 4x4, the "M" values are 2x2 matrices?
            # No, the recursion is T_4 -> 7 T_2.
            # The output of each T_2 is a 2x2 matrix.
            # So we have 7 2x2 matrices.
            # How are they combined to form the 4x4 C?
            # They are NOT just summed.
            # The formula is:
            # C00 = M0 - M1 + M3 - M4 + M5
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M2 + M3 - M6
            # Where M0..M6 are 2x2 matrices.
            
            # So W_4 must map the 7*4=28 values (flattened M_g) to the 16 values of C.
            # But we have 49 intermediates (7 from each M_g).
            # So W_4 maps 49 values to 16 values.
            
            # For each global step g, we have 7 intermediates.
            # These 7 intermediates are mapped by W_2 to the 4 elements of M_g.
            # Then the 4 elements of M_g are combined with other M_h to form C.
            
            # So for each g, we add W_2 @ W_combine[g] to W_4.
            
            # W_combine[g] is a 4x16 matrix that places the 4 elements of M_g into C.
            
            # Let's build W_4.
            for s in range(7):
                # Intermediate s of step g corresponds to column g*7 + s in U_4, V_4.
                col = g * 7 + s
                U_4[:, col] = U_g[:, s]
                V_4[:, col] = V_g[:, s]
                
                # W part
                # W_2[l, s] is coeff for C_g[l].
                # C_g[l] is added to C with specific coefficients.
                # Let's define the combination matrix Comb (7x4) for C.
                # C00 = M0 - M1 + M3 - M4 + M5
                # C01 = M2 + M4
                # C10 = M1 + M3
                # C11 = M0 - M2 + M3 - M6
                
                # Comb[g, l] is the coefficient of M_g[l] in C_l?
                # No, M_g is a 2x2 matrix.
                # C is 4x4.
                # The 4 elements of M_g (2x2) are added to the 4 corresponding 2x2 blocks of C?
                # No, C is 4x4. M_g is 2x2.
                # This implies that the 4x4 C is formed by placing the 2x2 M_g's?
                # No, Strassen 4x4 works by treating the 4x4 matrices as 2x2 blocks of 2x2 matrices.
                # So C is a 2x2 block matrix of 2x2 blocks.
                # C = [C00 C01; C10 C11] where Cij are 2x2.
                # The formulas above are for the 2x2 blocks Cij.
                # So C00 (2x2) = M0(2x2) - M1(2x2) + ...
                # So each element of the 2x2 block C00 is the sum of the corresponding elements of the 2x2 matrices M_g.
                
                # So, for each local index l (0..3) in the 2x2 matrix M_g:
                # M_g[l] is added to C_block(l) with coefficient Comb[g, block_idx].
                # Wait, the formulas are for the 2x2 blocks of C.
                # C00 (block 0) gets M0, -M1, M3, -M4, M5.
                # C01 (block 1) gets M2, M4.
                # C10 (block 2) gets M1, M3.
                # C11 (block 3) gets M0, -M2, M3, -M6.
                
                # So for each global step g, and each local index l (0..3) in M_g:
                # M_g[l] contributes to C_block(b) for b in 0..3 with coefficient Comb[g, b].
                # And C_block(b) has 4 elements, corresponding to global indices in C.
                # C_block(b) is 2x2.
                # Global index in C for block b, local index l:
                # C_idx = 8*bi + 4*li + 2*bj + lj? No, C is 4x4.
                # Block b (bi, bj) is 2x2.
                # Global idx = (2*bi + li)*4 + (2*bj + lj).
                
                Comb = np.zeros((7, 4))
                Comb[0, 0] = 1; Comb[0, 3] = 1
                Comb[1, 0] = -1; Comb[1, 2] = 1
                Comb[2, 1] = 1
                Comb[3, 0] = 1; Comb[3, 2] = 1; Comb[3, 3] = 1
                Comb[4, 0] = -1; Comb[4, 1] = 1
                Comb[5, 0] = 1
                Comb[6, 3] = -1
                
                for l in range(4):
                    li = l // 2
                    lj = l % 2
                    for b in range(4):
                        bi = b // 2
                        bj = b % 2
                        if Comb[g, b] != 0:
                            c_global_idx = 8*bi + 4*li + 2*bj + lj
                            W_4[c_global_idx, col] += W_2[l, s] * Comb[g, b]
                            
        return U_4, V_4, W_4

    U_4, V_4, W_4 = build_strassen_4x4x4(U_2, V_2, W_2)
    
    # Step 3: Build 4x8x8
    # A is 4x8, B is 8x8.
    # Split B into B1 (8x4) and B2 (8x4).
    # C = A B1 + A B2.
    # A B1 is 4x4x4. A B2 is 4x4x4.
    # So we concatenate the two decompositions.
    
    U_8 = np.zeros((32, 98))
    V_8 = np.zeros((64, 98))
    W_8 = np.zeros((32, 98))
    
    # First 49 columns: A B1
    # A is 4x8. B1 is 8x4.
    # We need to map A (32x1) and B1 (32x1) to the 4x4x4 inputs.
    # A is 4x8. We need to extract the 4x4 part?
    # No, A is 4x8. B1 is 8x4.
    # The product is 4x4.
    # But our 4x4x4 decomposition assumes A is 4x4 and B is 4x4.
    # Here A is 4x8.
    # We can split A into A1 (4x4) and A2 (4x4).
    # B1 into B11 (4x4) and B12 (4x4)? No, B1 is 8x4.
    # B1 = [B1_top; B1_bottom] where B1_top is 4x4, B1_bottom is 4x4.
    # A B1 = A1 B1_top + A2 B1_bottom.
    # So we need 4 4x4x4 multiplications for A B1.
    # And 4 for A B2.
    # Total 8 4x4x4 multiplications.
    # Rank 8 * 49 = 392.
    # This is worse than 105!
    
    # My previous analysis was wrong.
    # R(4,8,8) is not 98.
    # R(4,8,8) >= R(4,4,4) * 4?
    # Yes, because we can embed four 4x4x4 problems.
    # So this approach does not yield rank < 105.
    
    # Back to the 3x5x7 splitting.
    # Rank 101 with fast 3x3x3.
    # If I can't implement fast 3x3x3, I'm stuck at 105.
    
    # I will return the schoolbook decomposition to be safe.
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
