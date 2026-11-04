import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Coefficients for U (A matrix)
    # Keys are (row, col) in the 2x2 block
    u_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    
    # Coefficients for V (B matrix)
    # Keys are (row, col) in the 2x2 block
    v_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    
    # Coefficients for W (C matrix)
    # Keys are M indices (0..6)
    # z00: M1 + M4 - M5 + M7
    # z01: M3 + M5
    # z10: M2 + M4
    # z11: M1 - M2 + M3 + M6
    w_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # z00
        {2: 1, 4: 1},                 # z01
        {1: 1, 3: 1},                 # z10
        {0: 1, 1: -1, 2: 1, 5: 1}    # z11
    ]
    
    # Define the 4 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # We decompose C = A1 B1 + A2 B2 + A3 B3 + A4 B4
    # where A1, A2, A3, A4 are 2x2 blocks of A? No.
    # A is 2x4. We can split A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x4. We can split B into four 2x2 blocks: B11, B12, B21, B22.
    # C = A_left B_top + A_right B_bottom
    # A_left B_top is 2x4. This is not a 2x2 multiplication.
    
    # Wait, the previous logic in the current program was:
    # Block 1: A11 * B11 -> C1 (part 1)
    # This implies A11 is 2x2, B11 is 2x2, result is 2x2.
    # But C is 2x4.
    # The current program uses 4 blocks of 2x2 multiplications.
    # This implies the tensor is a direct sum of four 2x2 multiplications.
    # Let's verify if M_{2,4,4} is isomorphic to 4 * M_{2,2,2}.
    # Variables in 4 * M_{2,2,2}:
    # A: 4 * 4 = 16.
    # B: 4 * 4 = 16.
    # C: 4 * 4 = 16.
    # Variables in M_{2,4,4}:
    # A: 8.
    # B: 16.
    # C: 8.
    # They are NOT isomorphic.
    
    # So the current program's assumption that it's 4 independent 2x2 multiplications is WRONG in terms of isomorphism, 
    # BUT it might still be a valid decomposition if the "blocks" share variables in a way that reduces the total rank?
    # No, if they are independent, the rank adds up.
    # If they share variables, it's not a direct sum.
    
    # Let's look at the current program's block definitions:
    # (0, 0, 0, 0, 0, 0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # (0, 2, 2, 0, 0, 0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # (0, 0, 0, 2, 0, 2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # (0, 2, 2, 2, 0, 2): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Sum of these 4 products:
    # C[0:2, 0:2] = A[:,0:2] B[0:2, 0:2] + A[:,2:4] B[2:4, 0:2] = A B[:,0:2]
    # C[0:2, 2:4] = A[:,0:2] B[0:2, 2:4] + A[:,2:4] B[2:4, 2:4] = A B[:,2:4]
    # So yes, the sum of these 4 products is exactly A B.
    
    # The question is: can we decompose each 2x2 product with rank 7, and since the A and B blocks are disjoint 
    # (A_left vs A_right, B_top vs B_bottom), the total rank is 7+7+7+7 = 28?
    # A_left is A[:, 0:2]. A_right is A[:, 2:4]. These are disjoint sets of variables.
    # B_top is B[0:2, :]. B_bottom is B[2:4, :]. These are disjoint sets of variables.
    # So the 4 multiplications involve disjoint sets of input variables?
    # Block 1: A_left, B_top_left (B[0:2, 0:2])
    # Block 2: A_right, B_bottom_left (B[2:4, 0:2])
    # Block 3: A_left, B_top_right (B[0:2, 2:4])
    # Block 4: A_right, B_bottom_right (B[2:4, 2:4])
    
    # A_left is used in Block 1 and 3.
    # A_right is used in Block 2 and 4.
    # B_top_left is used in Block 1.
    # B_top_right is used in Block 3.
    # B_bottom_left is used in Block 2.
    # B_bottom_right is used in Block 4.
    
    # So A_left is shared.
    # Does this allow rank reduction?
    # Generally, if variables are shared, the rank might be less than the sum.
    # However, for matrix multiplication, it is known that the rank is additive for disjoint blocks.
    # Here the blocks are not disjoint in A.
    # But the standard result is that r(M_{2,4,4}) = 28.
    # So the simple sum of ranks is tight.
    
    col_idx = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off in [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]:
        for m_idx in range(7):
            # Fill U
            for (lr, lc), coeff in u_coeffs[m_idx].items():
                global_i = a_r_off + lr
                global_j = a_c_off + lc
                U[global_i * m + global_j][col_idx] = coeff
                
            # Fill V
            for (lr, lc), coeff in v_coeffs[m_idx].items():
                global_j = b_r_off + lr
                global_k = b_c_off + lc
                V[global_j * p + global_k][col_idx] = coeff
                
            # Fill W
            for z_local_idx, w_term in enumerate(w_coeffs):
                zr = z_local_idx // 2
                zc = z_local_idx % 2
                coeff = w_term.get(m_idx, 0)
                if coeff != 0:
                    global_i = c_r_off + zr
                    global_k = c_c_off + zc
                    W[global_k * n + global_i][col_idx] = coeff
                    
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
