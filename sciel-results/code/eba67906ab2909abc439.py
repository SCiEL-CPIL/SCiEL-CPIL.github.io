"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using block Strassen decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Block Strassen for 4x4:
    # Split into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, decomposed via scalar Strassen into 7 terms.
    
    # Scalar Strassen for 2x2:
    # Given X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
    # Z = XY:
    # s1 = (x00+x11)(y00+y11)
    # s2 = (x10+x11)y00
    # s3 = x00(y01-y11)
    # s4 = x11(y10-y00)
    # s5 = (x00+x01)y11
    # s6 = (x10-x00)(y00+y01)
    # s7 = (x01-x11)(y10+y11)
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # Now, for each block-level term M_t, the left and right "matrices" are 
    # linear combinations of A and B blocks.
    
    # Let me define the 49 terms.
    
    # For each block-level Strassen term t (0-6) and scalar Strassen term u (0-6),
    # we have a scalar multiplication.
    
    # The left factor for M_t is a linear combination of A blocks.
    # The right factor for M_t is a linear combination of B blocks.
    
    # For scalar Strassen, the left factor is a linear combination of the 4 elements
    # of the left 2x2 matrix, and the right factor is a linear combination of the 4
    # elements of the right 2x2 matrix.
    
    # Let me define the coefficients.
    
    # Block-level left factors (as linear combinations of A blocks):
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # Block-level right factors (as linear combinations of B blocks):
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # For each block-level term, the left "2x2 matrix" is actually a 2x2 matrix
    # where each element is a linear combination of A block elements.
    
    # Wait, I need to be more careful. Each A block is a 2x2 matrix. So A00 is a
    # 2x2 matrix with elements A00[0,0], A00[0,1], A00[1,0], A00[1,1].
    
    # When we say M1 = (A00+A11)(B00+B11), the left factor (A00+A11) is a 2x2
    # matrix where each element is the sum of the corresponding elements of A00 and A11.
    
    # So for scalar Strassen, x00 = A00[0,0] + A11[0,0], etc.
    
    # Let me define the 49 terms.
    
    # I'll use a helper to add a term.
    
    term_idx = 0
    
    # Define block-level Strassen terms
    # Each is (left_blocks, right_blocks) where left_blocks and right_blocks are
    # dicts mapping block indices to coefficients.
    
    block_terms = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # M3: A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M4: A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Scalar Strassen terms: for a 2x2 matrix X with elements x00,x01,x10,x11
    # and Y with elements y00,y01,y10,y11:
    # s1: x = (1,0,0,1), y = (1,0,0,1)  [x00+x11, y00+y11]
    # s2: x = (0,0,1,1), y = (1,0,0,0)  [x10+x11, y00]
    # s3: x = (1,0,0,0), y = (0,1,0,-1) [x00, y01-y11]
    # s4: x = (0,0,0,1), y = (0,0,1,-1) [x11, y10-y00]
    # s5: x = (1,1,0,0), y = (0,0,0,1)  [x00+x01, y11]
    # s6: x = (0,0,1,-1), y = (1,1,0,0) [x10-x00, y00+y01]
    # s7: x = (0,1,0,-1), y = (0,0,1,1) [x01-x11, y10+y11]
    
    scalar_terms = [
        ((1,0,0,1), (1,0,0,1)),   # s1
        ((0,0,1,1), (1,0,0,0)),   # s2
        ((1,0,0,0), (0,1,0,-1)),  # s3
        ((0,0,0,1), (0,0,1,-1)),  # s4
        ((1,1,0,0), (0,0,0,1)),   # s5
        ((0,0,1,-1), (1,1,0,0)),  # s6
        ((0,1,0,-1), (0,0,1,1)),  # s7
    ]
    
    # For each block-level term and scalar-level term, we create a scalar multiplication.
    
    for t, (left_blocks, right_blocks) in enumerate(block_terms):
        for u, (x_coeff, y_coeff) in enumerate(scalar_terms):
            # x_coeff = (x00_coeff, x01_coeff, x10_coeff, x11_coeff)
            # y_coeff = (y00_coeff, y01_coeff, y10_coeff, y11_coeff)
            
            # The left factor is a 2x2 matrix where:
            # element (i,j) = sum over blocks (a,b) of left_blocks[(a,b)] * A[a*2+i, b*2+j]
            
            # So for scalar Strassen element x00 (which is element (0,0) of the left 2x2):
            # x00 = sum over (a,b) in left_blocks of left_blocks[(a,b)] * A[a*2+0, b*2+0]
            
            # Similarly for x01, x10, x11.
            
            # Let me compute the U, V contributions.
            
            # U contribution: for each (a,b) in left_blocks, and for each (i,j) in {0,1}x{0,1},
            # the coefficient of A[a*2+i, b*2+j] in the left factor's (i,j) element is left_blocks[(a,b)].
            
            # But we need to combine with the scalar Strassen coefficients.
            
            # The left 2x2 matrix has elements:
            # L[0,0] = sum_{(a,b)} left_blocks[(a,b)] * A[a*2+0, b*2+0]
            # L[0,1] = sum_{(a,b)} left_blocks[(a,b)] * A[a*2+0, b*2+1]
            # L[1,0] = sum_{(a,b)} left_blocks[(a,b)] * A[a*2+1, b*2+0]
            # L[1,1] = sum_{(a,b)} left_blocks[(a,b)] * A[a*2+1, b*2+1]
            
            # The scalar Strassen left factor is:
            # s_left = x_coeff[0]*L[0,0] + x_coeff[1]*L[0,1] + x_coeff[2]*L[1,0] + x_coeff[3]*L[1,1]
            
            # So the total coefficient for A[a*2+i, b*2+j] is:
            # left_blocks[(a,b)] * x_coeff[i*2+j]
            
            # Wait, x_coeff is indexed as (x00, x01, x10, x11) which corresponds to (0,0), (0,1), (1,0), (1,1).
            
            # So for A element at position (a*2+i, b*2+j), the coefficient is:
            # left_blocks[(a,b)] * x_coeff[i*2+j]
            
            # Similarly for V:
            # For B element at position (c*2+i, d*2+j), the coefficient is:
            # right_blocks[(c,d)] * y_coeff[i*2+j]
            
            # Now for W: the result of the scalar multiplication contributes to the
            # block-level Strassen result M_t, which then contributes to C blocks.
            
            # The scalar Strassen result s_u contributes to the 2x2 result matrix Z as:
            # Z[0,0] = s1 + s4 - s5 + s7
            # Z[0,1] = s3 + s5
            # Z[1,0] = s2 + s4
            # Z[1,1] = s1 - s2 + s3 + s6
            
            # So for scalar term u, the contribution to Z[i,j] is:
            # z_coeff[u][i*2+j]
            
            # Then M_t contributes to C blocks as:
            # M1: C00 += 1, C11 += 1
            # M2: C10 += 1, C11 -= 1
            # M3: C01 += 1, C11 += 1
            # M4: C00 += 1, C10 += 1
            # M5: C00 -= 1, C01 += 1
            # M6: C11 += 1
            # M7: C00 += 1
            
            # Wait, let me re-check:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So M1 contributes to C00 with +1 and C11 with +1
            # M2 contributes to C10 with +1 and C11 with -1
            # M3 contributes to C01 with +1 and C11 with +1
            # M4 contributes to C00 with +1 and C10 with +1
            # M5 contributes to C00 with -1 and C01 with +1
            # M6 contributes to C11 with +1
            # M7 contributes to C00 with +1
            
            block_contrib = {
                0: {(0,0): 1, (1,1): 1},      # M1
                1: {(1,0): 1, (1,1): -1},     # M2
                2: {(0,1): 1, (1,1): 1},      # M3
                3: {(0,0): 1, (1,0): 1},      # M4
                4: {(0,0): -1, (0,1): 1},     # M5
                5: {(1,1): 1},                # M6
                6: {(0,0): 1},                # M7
            }
            
            # Scalar Strassen result coefficients:
            # Z[0,0] = s1 + s4 - s5 + s7
            # Z[0,1] = s3 + s5
            # Z[1,0] = s2 + s4
            # Z[1,1] = s1 - s2 + s3 + s6
            scalar_result_coeff = [
                (1, 0, 0, 1),   # s1: Z00+=1, Z01+=0, Z10+=0, Z11+=1
                (0, 0, 1, -1),  # s2: Z00+=0, Z01+=0, Z10+=1, Z11+=-1
                (0, 1, 0, 1),   # s3: Z00+=0, Z01+=1, Z10+=0, Z11+=1
                (1, 0, 1, 0),   # s4: Z00+=1, Z01+=0, Z10+=1, Z11+=0
                (-1, 1, 0, 0),  # s5: Z00+=-1, Z01+=1, Z10+=0, Z11+=0
                (0, 0, 0, 1),   # s6: Z00+=0, Z01+=0, Z10+=0, Z11+=1
                (1, 0, 0, 1),   # s7: Z00+=1, Z01+=0, Z10+=0, Z11+=1
            ]
            
            # Now compute U, V, W for this term.
            
            # U: for each A element
            for (a, b), a_coeff in left_blocks.items():
                for i in range(2):
                    for j in range(2):
                        coeff = a_coeff * x_coeff[i*2+j]
                        if coeff != 0:
                            U[a*2+i, b*2+j, term_idx] += coeff
            
            # V: for each B element
            for (c, d), b_coeff in right_blocks.items():
                for i in range(2):
                    for j in range(2):
                        coeff = b_coeff * y_coeff[i*2+j]
                        if coeff != 0:
                            V[c*2+i, d*2+j, term_idx] += coeff
            
            # W: for each C element
            # The scalar result Z[i,j] contributes to C blocks via block_contrib[t]
            z_coeff = scalar_result_coeff[u]
            for (ca, cb), c_block_coeff in block_contrib[t].items():
                for i in range(2):
                    for j in range(2):
                        coeff = c_block_coeff * z_coeff[i*2+j]
                        if coeff != 0:
                            # C element at position (ca*2+i, cb*2+j)
                            # W index: k*n + i where k is column, i is row
                            # So W[cb*2+j * 4 + ca*2+i, term_idx] += coeff
                            W[cb*2+j * 4 + ca*2+i, term_idx] += coeff
            
            term_idx += 1
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
