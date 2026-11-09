import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (20, rank)
    # W: (p*n, rank) -> (20, rank)
    
    # Rank 49 for 4x4x4 part
    # Rank 16 for 4x4x1 part
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for first 4 columns of B ---
    # Indices for A: i (0..3), j (0..3) -> idx = i*4 + j
    # Indices for B (first 4 cols): j (0..3), k (0..3) -> idx = j*5 + k
    # Indices for C (first 4 cols): k (0..3), i (0..3) -> idx = k*4 + i
    
    # Base Strassen for 2x2
    # Inputs: X (2x2), Y (2x2)
    # Outputs: Z (2x2)
    # 7 terms.
    # Term 1: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    # Term 2: (X10+X11)Y00 -> Z10 += 1
    # Term 3: X00(Y01-Y11) -> Z01 += 1
    # Term 4: X11(Y10-Y00) -> Z10 += 1, Z11 -= 1
    # Term 5: (X00+X01)Y11 -> Z01 += 1, Z11 -= 1
    # Term 6: (X10-X00)(Y00+Y01) -> Z10 += 1
    # Term 7: (X01-X11)(Y10+Y11) -> Z00 += 1, Z11 -= 1
    
    # Wait, standard Strassen recurrences:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the coefficients for X and Y for each of the 7 M's in the 4x4 context.
    # X and Y are 2x2 matrices formed from the 4x4 A and B.
    # We need to map the 2x2 indices (r, c) of X/Y to the 4x4 indices of A/B.
    # X is composed of blocks of A.
    # Block (br, bc) of A is A[2*br : 2*br+2, 2*bc : 2*bc+2].
    
    # For each Strassen step k (0..6), we define X_k and Y_k as 2x2 matrices of linear forms.
    # Instead of symbolic, we can just iterate.
    
    strassen_terms = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # X_coeffs: 4 entries (X00, X01, X10, X11) as lists of (block_row, block_col, coeff)
        # Actually, easier: define the 4x4 coefficient matrix for A and B for each X_k and Y_k.
    ]
    
    # Let's generate the 4x4 coefficient matrices for A and B for each of the 7 Strassen products.
    # A is 4x4. B is 4x4 (first 4 cols).
    # Block indices: (0,0), (0,1), (1,0), (1,1).
    # A_blocks:
    # A00: A[0:2, 0:2]
    # A01: A[0:2, 2:4]
    # A10: A[2:4, 0:2]
    # A11: A[2:4, 2:4]
    
    # We define 7 pairs of 4x4 matrices CA, CB such that the product of the corresponding 2x2 block matrices is the Strassen term.
    # Actually, we can just define the linear combination for the 4 entries of the 2x2 matrix X_k.
    
    # X_k is 2x2.
    # X_k[0,0], X_k[0,1], X_k[1,0], X_k[1,1].
    # Each is a sum of elements from A.
    
    def get_block_coeffs(block_r, block_c, sign=1):
        # Returns list of (i, j, coeff) for the 4x4 matrix A
        coeffs = []
        for dr in range(2):
            for dc in range(2):
                i = 2 * block_r + dr
                j = 2 * block_c + dc
                coeffs.append((i, j, sign))
        return coeffs

    # Define the 7 Strassen steps
    # Step 0: M1 = (A00 + A11)(B00 + B11)
    # X0 = A00 + A11
    # Y0 = B00 + B11
    steps = [
        # (X_blocks, Y_blocks)
        # X_blocks is list of (br, bc, coeff)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),           # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),          # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),          # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # M7
    ]
    
    # Recurrence coefficients for C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    recurrence = [
        # (block_r, block_c, coeff, m_index)
        (0, 0, 1, 0), (0, 0, 1, 3), (0, 0, -1, 4), (0, 0, 1, 6),
        (0, 1, 1, 2), (0, 1, 1, 4),
        (1, 0, 1, 1), (1, 0, 1, 3),
        (1, 1, 1, 0), (1, 1, -1, 1), (1, 1, 1, 2), (1, 1, 1, 5)
    ]
    
    # Base Strassen 2x2 terms
    # Each term t (0..6) has:
    # u_coeffs: 4 coeffs for X[0,0], X[0,1], X[1,0], X[1,1]
    # v_coeffs: 4 coeffs for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
    # w_coeffs: 4 coeffs for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
    
    base_terms = [
        # T0: (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # T1: (X10+X11)Y00 -> Z10+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # T2: X00(Y01-Y11) -> Z01+=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # T3: X11(Y10-Y00) -> Z10+=1, Z11-=1
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, -1]),
        # T4: (X00+X01)Y11 -> Z01+=1, Z11-=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
        # T5: (X10-X00)(Y00+Y01) -> Z10+=1
        ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]), # Note: X10-X00 is [ -1, 0, 1, 0 ]? 
        # Let's re-verify T5.
        # M6 = (A10 - A00)(B00 + B01)
        # X10 - X00. Coeffs: X00:-1, X10:1. So [-1, 0, 1, 0].
        # Y00 + Y01. Coeffs: Y00:1, Y01:1. So [1, 1, 0, 0].
        # Result Z10 += 1.
        # So T5 should be:
        # u: [-1, 0, 1, 0]
        # v: [1, 1, 0, 0]
        # w: [0, 0, 1, 0]
        
        # T6: (X01-X11)(Y10+Y11) -> Z00+=1, Z11-=1
        # X01 - X11. Coeffs: X01:1, X11:-1. So [0, 1, 0, -1].
        # Y10 + Y11. Coeffs: Y10:1, Y11:1. So [0, 0, 1, 1].
        # Result Z00 += 1, Z11 -= 1.
        # So T6:
        # u: [0, 1, 0, -1]
        # v: [0, 0, 1, 1]
        # w: [1, 0, 0, -1]
    ]
    
    # Correcting T5 in base_terms list
    base_terms[5] = ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0])
    base_terms[6] = ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    
    col = 0
    for step_idx, (x_blocks, y_blocks) in enumerate(steps):
        # Construct 4x4 coefficient matrices for A and B for this step
        # A_coeffs[i][j] = sum of coeffs from blocks
        A_coeff = [[0]*4 for _ in range(4)]
        B_coeff = [[0]*4 for _ in range(4)]
        
        for br, bc, coeff in x_blocks:
            for dr in range(2):
                for dc in range(2):
                    i = 2*br + dr
                    j = 2*bc + dc
                    A_coeff[i][j] += coeff
                    
        for br, bc, coeff in y_blocks:
            for dr in range(2):
                for dc in range(2):
                    i = 2*br + dr
                    j = 2*bc + dc
                    B_coeff[i][j] += coeff
                    
        # Now apply base Strassen terms
        for t_idx, (u_c, v_c, w_c) in enumerate(base_terms):
            # u_c corresponds to X[0,0], X[0,1], X[1,0], X[1,1]
            # We need to map X indices to A indices.
            # X[0,0] is the top-left 2x2 block of the "virtual" 2x2 matrix X.
            # Wait, X is a 2x2 matrix of scalars? No.
            # In the Strassen algorithm for 4x4, X_k is a 2x2 matrix where each entry is a scalar?
            # No. X_k is a 2x2 matrix formed by linear combinations of the 2x2 blocks.
            # So X_k[0,0] is a scalar? No, X_k[0,0] is a 2x2 matrix?
            # Let's re-read Strassen.
            # A is 4x4. A11 is 2x2.
            # M1 = (A11 + A22)(B11 + B22).
            # A11 + A22 is a 2x2 matrix.
            # So the input to the "multiplication" is a 2x2 matrix.
            # So we are decomposing the multiplication of two 2x2 matrices.
            # So u_c coefficients apply to the 4 elements of the 2x2 matrix X_k.
            # The elements of X_k are X_k[0,0], X_k[0,1], X_k[1,0], X_k[1,1].
            # These are scalars? No, they are the entries of the 2x2 matrix.
            # But the 2x2 matrix X_k is composed of the blocks of A.
            # So X_k[0,0] is the (0,0) entry of the 2x2 matrix X_k.
            # This entry is a linear combination of the entries of the blocks?
            # No. X_k is a 2x2 matrix. Its entries are scalars.
            # But A is a 4x4 matrix of scalars.
            # So A11 is a 2x2 submatrix.
            # (A11 + A22) is a 2x2 matrix.
            # Its (0,0) entry is A[0,0] + A[2,2].
            # So X_k[0,0] = A[0,0] + A[2,2].
            # So when we say u_c[0] (coeff for X[0,0]), it applies to the scalar value X_k[0,0].
            # And X_k[0,0] is a linear combination of A's elements.
            # So the total coeff for A[i,j] is u_c[x_idx] * (coeff of A[i,j] in X_k[x_idx]).
            
            # Let's build the mapping from X_k indices (0..3) to A indices (0..15).
            # X_k[0,0] corresponds to block (0,0) entry (0,0) + block (1,1) entry (0,0)?
            # No. X_k is the 2x2 matrix resulting from the block sum.
            # So X_k[0,0] is the (0,0) element of that 2x2 matrix.
            # That element is A[0,0] + A[2,2].
            # X_k[0,1] is A[0,1] + A[2,3].
            # X_k[1,0] is A[1,0] + A[3,2].
            # X_k[1,1] is A[1,1] + A[3,3].
            
            # So we need to map x_idx (0..3) to the 4 A elements.
            # x_idx 0 (X[0,0]): A[0,0], A[2,2]
            # x_idx 1 (X[0,1]): A[0,1], A[2,3]
            # x_idx 2 (X[1,0]): A[1,0], A[3,2]
            # x_idx 3 (X[1,1]): A[1,1], A[3,3]
            
            # This mapping depends on the specific step (which blocks are added).
            # We already have A_coeff[i][j] which is the coeff of A[i,j] in the 2x2 matrix X_k?
            # No. A_coeff[i][j] is the coeff of A[i,j] in the "block matrix" X_k.
            # Wait. X_k is a 2x2 matrix.
            # The entries of X_k are scalars.
            # My A_coeff is 4x4.
            # This implies I am treating X_k as a 4x4 matrix?
            # No. The multiplication is of 2x2 matrices.
            # So we need to decompose the 2x2 matrix multiplication.
            # The inputs are 2x2 matrices X_k and Y_k.
            # The elements of X_k are linear combinations of A's elements.
            
            # Let's define the 4 elements of X_k explicitly.
            # X_k[0,0], X_k[0,1], X_k[1,0], X_k[1,1].
            # We can compute the coefficients for each of these 4 scalars in terms of A's 16 elements.
            
            # Let's re-construct A_coeff as 4 lists of 16 coeffs.
            # a_coeffs[x_idx][i, j]
            
            a_coeffs = [[0]*16 for _ in range(4)]
            b_coeffs = [[0]*16 for _ in range(4)]
            
            # Map x_idx to (block_r, block_c, local_r, local_c)
            # x_idx 0: (0,0)
            # x_idx 1: (0,1)
            # x_idx 2: (1,0)
            # x_idx 3: (1,1)
            
            for x_idx in range(4):
                br = x_idx // 2
                bc = x_idx % 2
                # Find which blocks contribute to X_k[br, bc]
                # From x_blocks: list of (br, bc, coeff)
                # If (br, bc) is in x_blocks, then the block contributes.
                for block_br, block_bc, coeff in x_blocks:
                    if block_br == br and block_bc == bc:
                        # The block (block_br, block_bc) contributes with coeff.
                        # The block is 2x2.
                        # Its elements map to A indices.
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*block_br + dr
                                j = 2*block_bc + dc
                                a_coeffs[x_idx][i*4+j] += coeff
            
            # Similarly for B
            for y_idx in range(4):
                br = y_idx // 2
                bc = y_idx % 2
                for block_br, block_bc, coeff in y_blocks:
                    if block_br == br and block_bc == bc:
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*block_br + dr
                                j = 2*block_bc + dc
                                b_coeffs[y_idx][i*4+j] += coeff
            
            # Now, for this base term t_idx:
            # u_c has 4 coeffs for X[0,0]..X[1,1]
            # Total coeff for A[i,j] is sum_{x_idx} u_c[x_idx] * a_coeffs[x_idx][i*4+j]
            
            total_a = [0]*16
            for i in range(16):
                val = 0
                for x_idx in range(4):
                    val += u_c[x_idx] * a_coeffs[x_idx][i]
                total_a[i] = val
                
            total_b = [0]*16
            for i in range(16):
                val = 0
                for y_idx in range(4):
                    val += v_c[y_idx] * b_coeffs[y_idx][i]
                total_b[i] = val
            
            # Fill U and V
            for i in range(16):
                U[i][col] = total_a[i]
                V[i][col] = total_b[i] # B indices for first 4 cols: i*5 + j? No.
                # V index is j*p + k. Here j is row of B, k is col.
                # Our B is 4x4 (first 4 cols).
                # i in 0..15 corresponds to row j=i//4, col k=i%4.
                # V index = j*5 + k.
                pass
                
            # Correct V indexing
            for i in range(16):
                j = i // 4
                k = i % 4
                V[j*5 + k][col] = total_b[i]
                
            # Fill W
            # w_c has 4 coeffs for Z[0,0]..Z[1,1]
            # Z is the 2x2 result of the 2x2 multiplication.
            # This Z contributes to the C blocks according to the recurrence.
            # We need to map Z indices (0..3) to C indices (0..15).
            
            # Z_idx 0: (0,0) -> C block (0,0)
            # Z_idx 1: (0,1) -> C block (0,1)
            # Z_idx 2: (1,0) -> C block (1,0)
            # Z_idx 3: (1,1) -> C block (1,1)
            
            total_c = [0]*16
            for z_idx in range(4):
                coeff = w_c[z_idx]
                if coeff == 0: continue
                br = z_idx // 2
                bc = z_idx % 2
                # Find which recurrence terms use M_step_idx (which is step_idx)
                # And contribute to block (br, bc)
                for r_br, r_bc, r_coeff, r_m_idx in recurrence:
                    if r_m_idx == step_idx and r_br == br and r_bc == bc:
                        # This Z element contributes to C block (r_br, r_bc) with coeff r_coeff
                        # The Z element is a 2x2 matrix? No, Z is 2x2.
                        # Z[z_idx] is a scalar?
                        # In the base Strassen, Z is a 2x2 matrix of scalars.
                        # So Z[z_idx] is the scalar at position (br, bc) of the 2x2 result.
                        # This scalar is added to the corresponding element of the C block.
                        # The C block is 2x2.
                        # So the scalar Z[z_idx] is added to all 4 elements of the C block?
                        # No. The base Strassen computes the 2x2 product.
                        # The result is a 2x2 matrix of scalars.
                        # Each scalar in the result corresponds to one element of the 2x2 block.
                        # So Z[z_idx] is the value for C_block[br, bc] at position (local_r, local_c)?
                        # No. The base Strassen operates on 2x2 matrices of scalars.
                        # So the input X is 2x2 scalars.
                        # The output Z is 2x2 scalars.
                        # So Z[0,0] is a scalar.
                        # This scalar is the (0,0) element of the 2x2 product.
                        # In the 4x4 context, this scalar is the (0,0) element of the C block?
                        # Yes.
                        # So Z[z_idx] contributes to C[2*br + dr, 2*bc + dc] for dr,dc in 0..1?
                        # No. Z is 2x2.
                        # Z[0,0] is the top-left element of the 2x2 result.
                        # This corresponds to the top-left element of the C block.
                        # So Z[0,0] contributes to C[0,0] of the block.
                        # Z[0,1] contributes to C[0,1] of the block.
                        # etc.
                        
                        # So for each local (dr, dc) in 2x2:
                        for dr in range(2):
                            for dc in range(2):
                                # Z[z_idx] is the value at (z_br, z_bc) of the 2x2 result.
                                # z_br = z_idx // 2, z_bc = z_idx % 2.
                                # This value is added to C[2*r_br + dr, 2*r_bc + dc] ?
                                # No. The 2x2 result Z is placed into the C block.
                                # So Z[z_br, z_bc] is the value for C[2*r_br + z_br, 2*r_bc + z_bc]?
                                # No. The block is 2x2.
                                # The result Z is 2x2.
                                # So Z[0,0] goes to C[0,0] of the block.
                                # Z[0,1] goes to C[0,1] of the block.
                                # Z[1,0] goes to C[1,0] of the block.
                                # Z[1,1] goes to C[1,1] of the block.
                                
                                # So Z[z_idx] contributes to C[2*r_br + z_br, 2*r_bc + z_bc].
                                # Wait. z_idx is the index in the 2x2 result.
                                # z_br = z_idx // 2.
                                # z_bc = z_idx % 2.
                                # The target C index is 2*r_br + z_br, 2*r_bc + z_bc.
                                
                                c_i = 2*r_br + z_br
                                c_j = 2*r_bc + z_bc
                                c_idx = c_j*4 + c_i # W index is k*n + i. k=c_j, i=c_i.
                                total_c[c_idx] += coeff * r_coeff
            
            for c_idx in range(16):
                W[c_idx][col] = total_c[c_idx]
                
            col += 1
            
    # --- Part 2: Schoolbook for 5th column ---
    # C_i,4 = sum_j A_i,j B_j,4
    # i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
