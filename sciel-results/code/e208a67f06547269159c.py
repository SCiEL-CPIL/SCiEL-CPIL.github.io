"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We'll build 4 Strassen decompositions for the 4 sub-multiplications
    # Each Strassen contributes 7 rank terms, total 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for a 2x2x2 multiplication
    # Products P1..P7 as linear combos of X elements (x00,x01,x10,x11) and Y elements (y00,y01,y10,y11)
    # X indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Y indices: 0=y00, 1=y01, 2=y10, 3=y11
    
    # For each product, we need coefficients for X and Y
    # P1 = (x00+x11)(y00+y11): X coeffs [1,0,0,1], Y coeffs [1,0,0,1]
    # P2 = (x10+x11)(y00): X coeffs [0,0,1,1], Y coeffs [1,0,0,0]
    # P3 = x00(y01-y11): X coeffs [1,0,0,0], Y coeffs [0,1,0,-1]
    # P4 = x11(y10-y00): X coeffs [0,0,0,1], Y coeffs [-1,0,1,0]
    # P5 = (x00+x01)(y11): X coeffs [1,1,0,0], Y coeffs [0,0,0,1]
    # P6 = (x10-x00)(y00+y01): X coeffs [-1,0,1,0], Y coeffs [1,1,0,0]
    # P7 = (x01-x11)(y10+y11): X coeffs [0,1,0,-1], Y coeffs [0,0,1,1]
    
    # Output coefficients: Z elements z00,z01,z10,z11 as combos of P1..P7
    # z00 = P1 + P4 - P5 + P7: [1,0,0,1,-1,0,1]
    # z01 = P3 + P5: [0,0,1,0,1,0,0]
    # z10 = P2 + P4: [0,1,0,1,0,0,0]
    # z11 = P1 - P2 + P3 + P6: [1,-1,1,0,0,1,0]
    
    strassen_X = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1], # P7
    ]
    
    strassen_Y = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1],  # P7
    ]
    
    strassen_Z = [
        [1, 0, 0, 1, -1, 0, 1],   # z00
        [0, 0, 1, 0, 1, 0, 0],    # z01
        [0, 1, 0, 1, 0, 0, 0],    # z10
        [1, -1, 1, 0, 0, 1, 0],   # z11
    ]
    
    # Now define the 4 sub-multiplications:
    # Each is: X_sub (2x2) * Y_sub (2x2) -> Z_sub (2x2)
    # We need to map local 2x2 indices to global A, B, C indices
    
    # Sub-multiplication 1: A1 * B11 -> C1_part1
    # A1 = A[:, 0:2]: local (a_row, a_col) -> global A index (a_row * 4 + a_col)
    #   local 0=(0,0)->A[0], 1=(0,1)->A[1], 2=(1,0)->A[4], 3=(1,1)->A[5]
    # B11 = B[0:2, 0:2]: local (b_row, b_col) -> global B index (b_row * 4 + b_col)
    #   local 0=(0,0)->B[0], 1=(0,1)->B[1], 2=(1,0)->B[4], 3=(1,1)->B[5]
    # C1_part1 contributes to C[:, 0:2]: local (c_row, c_col) -> global C index
    #   W index = k*n + i = c_col*2 + c_row
    #   local 0=(0,0)->W[0], 1=(0,1)->W[1], 2=(1,0)->W[2], 3=(1,1)->W[3]
    
    # Sub-multiplication 2: A2 * B21 -> C1_part2
    # A2 = A[:, 2:4]: local (a_row, a_col) -> global A index (a_row * 4 + 2 + a_col)
    #   local 0=(0,0)->A[2], 1=(0,1)->A[3], 2=(1,0)->A[6], 3=(1,1)->A[7]
    # B21 = B[2:4, 0:2]: local (b_row, b_col) -> global B index ((2+b_row) * 4 + b_col)
    #   local 0=(0,0)->B[8], 1=(0,1)->B[9], 2=(1,0)->B[12], 3=(1,1)->B[13]
    # Contributes to C[:, 0:2]: same W mapping as sub-mul 1
    
    # Sub-multiplication 3: A1 * B12 -> C2_part1
    # A1 = A[:, 0:2]: same as sub-mul 1
    # B12 = B[0:2, 2:4]: local (b_row, b_col) -> global B index (b_row * 4 + 2 + b_col)
    #   local 0=(0,0)->B[2], 1=(0,1)->B[3], 2=(1,0)->B[6], 3=(1,1)->B[7]
    # Contributes to C[:, 2:4]: local (c_row, c_col) -> W index = (2+c_col)*2 + c_row
    #   local 0=(0,0)->W[4], 1=(0,1)->W[5], 2=(1,0)->W[6], 3=(1,1)->W[7]
    
    # Sub-multiplication 4: A2 * B22 -> C2_part2
    # A2 = A[:, 2:4]: same as sub-mul 2
    # B22 = B[2:4, 2:4]: local (b_row, b_col) -> global B index ((2+b_row) * 4 + 2 + b_col)
    #   local 0=(0,0)->B[10], 1=(0,1)->B[11], 2=(1,0)->B[14], 3=(1,1)->B[15]
    # Contributes to C[:, 2:4]: same W mapping as sub-mul 3
    
    # Define index mappings for each sub-multiplication
    # For each sub-mul: list of 4 global A indices, 4 global B indices, 4 global W indices
    
    sub_muls = [
        # (A_indices, B_indices, W_indices)
        # Sub-mul 1: A1 * B11 -> C[0:2, 0:2]
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]),
        # Sub-mul 2: A2 * B21 -> C[0:2, 0:2]
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]),
        # Sub-mul 3: A1 * B12 -> C[2:4, 0:2]
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]),
        # Sub-mul 4: A2 * B22 -> C[2:4, 0:2]
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7]),
    ]
    
    for sub_idx in range(4):
        A_indices, B_indices, W_indices = sub_muls[sub_idx]
        base_col = sub_idx * 7
        
        for prod_idx in range(7):
            col = base_col + prod_idx
            
            # Fill U: for each local X element, add strassen_X[prod_idx][local] to U[global_A_idx][col]
            for local_x in range(4):
                U[A_indices[local_x]][col] += strassen_X[prod_idx][local_x]
            
            # Fill V: for each local Y element, add strassen_Y[prod_idx][local] to V[global_B_idx][col]
            for local_y in range(4):
                V[B_indices[local_y]][col] += strassen_Y[prod_idx][local_y]
            
            # Fill W: for each local Z element, add strassen_Z[local_z][prod_idx] to W[global_W_idx][col]
            for local_z in range(4):
                W[W_indices[local_z]][col] += strassen_Z[local_z][prod_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
