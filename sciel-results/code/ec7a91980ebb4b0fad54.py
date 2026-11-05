import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank_total = 97
    
    U = [[0] * rank_total for _ in range(n * m)] # 16 x 97
    V = [[0] * rank_total for _ in range(m * p)] # 28 x 97
    W = [[0] * rank_total for _ in range(p * n)] # 28 x 97
    
    col_offset = 0
    
    # Part 1: 4x4x4 Strassen
    # This handles B cols 0-3 and C cols 0-3
    # A is 4x4, B_part is 4x4, C_part is 4x4
    
    # Strassen 4x4x4 uses 7 Strassen 2x2x2 steps
    # Each 2x2x2 step uses 7 scalar multiplications
    
    # We need to define the 7 Strassen products for 4x4
    # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
    # B blocks: B11 (0:2, 0:2), B12 (0:2, 2:4), B21 (2:4, 0:2), B22 (2:4, 2:4)
    
    # Strassen steps for 4x4:
    # S1: (A11+A22)(B11+B22) -> C11 += S1, C22 -= S1 (wait, C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6)
    # Let's use the standard 4x4 Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product.
    # We will decompose each Mk using Strassen 2x2x2.
    
    # Helper to add a 2x2x2 Strassen decomposition to U,V,W
    # local_a_offset, local_b_offset, local_c_offset
    # a_signs, b_signs: coefficients for the 2x2 blocks of A and B in this step
    # c_signs: coefficients to add the result to the 2x2 blocks of C
    
    # Actually, it's easier to just iterate through the 7 Strassen steps and for each,
    # determine the linear combinations of A and B entries,
    # then use the 2x2x2 Strassen decomposition.
    
    # 2x2x2 Strassen decomposition details:
    # For A' (2x2) and B' (2x2):
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pk is a scalar product of two linear combinations?
    # No, Pk is a 2x2 matrix.
    # Wait, in 2x2x2 Strassen, the "multiplications" are scalar multiplications of the linear combinations.
    # Let's clarify:
    # P1 = (A11+A22) * (B11+B22) -> This is a scalar multiplication if A11 etc are scalars.
    # Yes, for 2x2x2, the blocks are scalars.
    
    # So for each 4x4 Strassen step Mk (which is a 2x2x2 multiplication of blocks),
    # we treat the blocks of A (2x2) and B (2x2) as the inputs to a 2x2x2 Strassen.
    
    # We need to map the global indices to the local 2x2 indices.
    
    # Let's define the 7 steps for 4x4x4.
    # Step k:
    # A_comb: 2x2 matrix of coefficients for A blocks
    # B_comb: 2x2 matrix of coefficients for B blocks
    # C_comb: 2x2 matrix of coefficients for C blocks (with signs)
    
    steps_4x4 = [
        # (A_coeff, B_coeff, C_coeff)
        # A_coeff is 2x2, B_coeff is 2x2, C_coeff is 2x2
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]), # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1? No.
        # C11 = M1 + ... C22 = M1 - ...
        # So C_coeff for M1 is [[1, 0], [0, 1]]?
        # C11 gets +1, C22 gets +1?
        # Formula: C11 = M1 + M4 - M5 + M7. C22 = M1 - M2 + M3 + M6.
        # So M1 contributes +1 to C11 and +1 to C22.
        # M2 contributes +1 to C21 and -1 to C22.
        # M3 contributes +1 to C12 and +1 to C22.
        # M4 contributes +1 to C11 and +1 to C21.
        # M5 contributes -1 to C11 and +1 to C12.
        # M6 contributes +1 to C22.
        # M7 contributes +1 to C11.
        
        # Let's re-verify Strassen 4x4 formulas.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C11+1, C22+1
        # M2: C21+1, C22-1
        # M3: C12+1, C22+1
        # M4: C11+1, C21+1
        # M5: C11-1, C12+1
        # M6: C22+1
        # M7: C11+1
        
        # A/B combinations:
        # M1: A=(1,0;0,1) (A11+A22), B=(1,0;0,1) (B11+B22)
        # M2: A=(0,0;1,1) (A21+A22), B=(1,0;0,0) (B11)
        # M3: A=(1,0;0,0) (A11), B=(0,1;0,-1) (B12-B22)
        # M4: A=(0,0;0,1) (A22), B=(0,-1;1,0) (B21-B11)
        # M5: A=(1,1;0,0) (A11+A12), B=(0,0;0,1) (B22)
        # M6: A=(0,-1;1,0) (A21-A11), B=(1,1;0,0) (B11+B12)
        # M7: A=(0,1;0,-1) (A12-A22), B=(0,0;1,1) (B21+B22)
    ]
    
    # We will process each of these 7 steps.
    # For each step, we have a 2x2x2 multiplication of blocks.
    # We decompose this 2x2x2 multiplication using Strassen (rank 7).
    
    for step_idx in range(7):
        A_coeff, B_coeff, C_coeff = steps_4x4[step_idx]
        
        # For this step, we add 7 columns to U, V, W.
        # These correspond to the 7 Strassen products P1..P7 for the 2x2x2 block multiplication.
        
        # The 2x2x2 Strassen products Pk are:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # And the contributions to C (2x2 blocks):
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We need to combine the outer coefficients (A_coeff, B_coeff) with the inner Strassen coefficients.
        
        # Let's define the inner Strassen coefficients for P1..P7.
        # A_inner is 2x2, B_inner is 2x2, C_inner is 2x2 (for the 2x2 block result)
        
        inner_steps = [
            # P1
            ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            # P2
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]), # P2 -> C21
            # P3
            ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]), # P3 -> C12
            # P4
            ([[0, 0], [0, 1]], [[0, -1], [1, 0]], [[1, 0], [1, 0]]), # P4 -> C11, C21
            # P5
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]), # P5 -> C11(-), C12(+)
            # P6
            ([[0, -1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]), # P6 -> C22
            # P7
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])  # P7 -> C11
        ]
        
        for inner_idx in range(7):
            A_in, B_in, C_in = inner_steps[inner_idx]
            
            # Current column
            col = col_offset
            col_offset += 1
            
            # Calculate effective A coefficients (4x4)
            # A_eff[i,j] = sum_{u,v} A_coeff[u,v] * A_in[u,v] * delta_{i, 2*u+row} delta_{j, 2*v+col}
            # This is getting complex. Let's just iterate.
            
            # For each of the 4 blocks of A (2x2 blocks):
            # Block (u,v) of A is at rows 2u:2u+2, cols 2v:2v+2.
            # The coefficient for this block in the product is A_coeff[u,v] * A_in[u,v].
            # Wait, A_in is the coefficient matrix for the 2x2x2 Strassen.
            # A_in[u,v] is the coefficient of the (u,v) block of the *local* A.
            # The *local* A is formed by A_coeff.
            # So the coefficient for the global A block (u,v) is A_coeff[u,v] * A_in[u,v].
            
            # Similarly for B.
            
            # And for C, the contribution to global C block (u,v) is C_coeff[u,v] * C_in[u,v].
            
            for u in range(2):
                for v in range(2):
                    coeff_a = A_coeff[u][v] * A_in[u][v]
                    if coeff_a != 0:
                        # Add to U for block (u,v)
                        # Block (u,v) of A is at rows 2u..2u+1, cols 2v..2v+1
                        for i in range(2):
                            for j in range(2):
                                U[2*u + i][2*v + j][col] = coeff_a
                            
                    coeff_b = B_coeff[u][v] * B_in[u][v]
                    if coeff_b != 0:
                        # Add to V for block (u,v) of B
                        # Block (u,v) of B is at rows 2u..2u+1, cols 2v..2v+1
                        # Note: B is 4x4 in this sub-problem.
                        # Global B rows 0-3, cols 0-3.
                        for i in range(2):
                            for j in range(2):
                                V[2*u + i][2*v + j][col] = coeff_b
                                
                    coeff_c = C_coeff[u][v] * C_in[u][v]
                    if coeff_c != 0:
                        # Add to W for block (u,v) of C
                        # Block (u,v) of C is at rows 2v..2v+1 (cols of C), cols 2u..2u+1 (rows of C)
                        # W index is k*n + i.
                        # k is col of C (0-3), i is row of C (0-3).
                        # Block (u,v) of C: rows 2u..2u+1, cols 2v..2v+1.
                        for i in range(2):
                            for k in range(2):
                                # Global C row = 2u + i, col = 2v + k
                                # W index = (2v + k) * 4 + (2u + i)
                                W[(2*v + k) * 4 + (2*u + i)][col] = coeff_c

    # Part 2: 4x4x3 Schoolbook
    # B cols 4-6, C cols 4-6
    # A is 4x4, B_part is 4x3, C_part is 4x3
    # Standard decomposition:
    # For each j in 0..3 (col of A / row of B):
    #   For each k in 4..6 (col of B / col of C):
    #     For each i in 0..3 (row of A / row of C):
    #       Term: A[i,j] * B[j,k] -> C[i,k]
    #       U[i,j] += 1, V[j,k] += 1, W[k,i] += 1
    
    for j in range(4):
        for k in range(3):
            global_k = k + 4
            for i in range(4):
                col = col_offset
                col_offset += 1
                U[i * 4 + j][col] = 1
                V[j * 7 + global_k][col] = 1
                W[global_k * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}

