"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
This uses a two-level Strassen algorithm:
1. Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
2. Apply Strassen's 7-multiplication algorithm at the block level.
3. Apply Strassen's 7-multiplication algorithm at the 2x2 matrix level for each of the 7 block-products.
Total multiplications: 7 * 7 = 49.
"""

import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-49 decomposition of 4x4 matrix multiplication.
    U, V, W are lists of lists with integer entries.
    Shapes: U (16, 49), V (16, 49), W (16, 49).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices A, B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A-side) and V (B-side) for each of the 7 Strassen terms
    # Indices are (row, col) in the 2x2 block
    # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    
    U_coeffs = [
        { (0,0): 1, (1,1): 1 },  # M1
        { (1,0): 1, (1,1): 1 },  # M2
        { (0,0): 1 },            # M3
        { (1,1): 1 },            # M4
        { (0,0): 1, (0,1): 1 },  # M5
        { (1,0): 1, (0,0): -1 }, # M6
        { (0,1): 1, (1,1): -1 }  # M7
    ]
    
    V_coeffs = [
        { (0,0): 1, (1,1): 1 },  # M1
        { (0,0): 1 },            # M2
        { (0,1): 1, (1,1): -1 }, # M3
        { (1,0): 1, (0,0): -1 }, # M4
        { (1,1): 1 },            # M5
        { (0,0): 1, (0,1): 1 },  # M6
        { (1,0): 1, (1,1): 1 }   # M7
    ]
    
    # W coefficients for 2x2 result C from M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_coeffs[c_row, c_col] is a list of 7 coeffs for M1..M7
    W_coeffs = [
        [ [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0] ], # C00, C01
        [ [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0] ]  # C10, C11
    ]

    # The decomposition is hierarchical.
    # Outer level: 7 terms (M1..M7) for 4x4 blocks (each block is 2x2 matrices)
    # Inner level: 7 terms (m1..m7) for 2x2 matrices
    
    # We iterate over the 49 combinations (outer_term, inner_term)
    col = 0
    for outer_t in range(7):
        # Outer term coefficients for A and B blocks
        # These are 2x2 blocks of 2x2 matrices.
        # U_outer[outer_t] is a dict of (block_row, block_col) -> coeff
        # But wait, the U_coeffs above are for the 2x2 matrix entries.
        # The outer Strassen operates on 2x2 blocks.
        # So for outer term t, we have linear combinations of A-blocks.
        
        # Let's define the outer coefficients explicitly for the 2x2 block structure.
        # A is 4x4, blocks A00, A01, A10, A11 are 2x2.
        # Outer M1 = (A00+A11)(B00+B11)
        # So for outer term 0 (M1), A part is A00 + A11.
        # This means for each element in the 2x2 sub-matrix of A00 and A11, we add 1.
        
        # It is easier to think of the full 4x4 matrix indices.
        # A[i, j] where i,j in 0..3.
        # Block A_br_bc contains A[2*br+i, 2*bc+j] for i,j in 0,1.
        
        # Outer Coefficients for A (4x4 matrix indices)
        # For each outer term t, U_outer is a 4x4 matrix of coefficients.
        U_outer_coeffs = []
        V_outer_coeffs = []
        
        # Helper to create 4x4 coeff matrix from block coeffs
        def expand_block_coeffs(block_coeffs):
            # block_coeffs is dict (br, bc) -> coeff
            mat = np.zeros((4, 4), dtype=int)
            for (br, bc), c in block_coeffs.items():
                for i in range(2):
                    for j in range(2):
                        mat[2*br+i, 2*bc+j] += c
            return mat
        
        # Outer Strassen Block Coefficients
        # M1: (A00+A11)
        U_outer_coeffs.append(expand_block_coeffs({(0,0):1, (1,1):1}))
        V_outer_coeffs.append(expand_block_coeffs({(0,0):1, (1,1):1}))
        
        # M2: (A10+A11)
        U_outer_coeffs.append(expand_block_coeffs({(1,0):1, (1,1):1}))
        V_outer_coeffs.append(expand_block_coeffs({(0,0):1}))
        
        # M3: A00
        U_outer_coeffs.append(expand_block_coeffs({(0,0):1}))
        V_outer_coeffs.append(expand_block_coeffs({(0,1):1, (1,1):-1}))
        
        # M4: A11
        U_outer_coeffs.append(expand_block_coeffs({(1,1):1}))
        V_outer_coeffs.append(expand_block_coeffs({(1,0):1, (0,0):-1}))
        
        # M5: (A00+A01)
        U_outer_coeffs.append(expand_block_coeffs({(0,0):1, (0,1):1}))
        V_outer_coeffs.append(expand_block_coeffs({(1,1):1}))
        
        # M6: (A10-A00)
        U_outer_coeffs.append(expand_block_coeffs({(1,0):1, (0,0):-1}))
        V_outer_coeffs.append(expand_block_coeffs({(0,0):1, (0,1):1}))
        
        # M7: (A01-A11)
        U_outer_coeffs.append(expand_block_coeffs({(0,1):1, (1,1):-1}))
        V_outer_coeffs.append(expand_block_coeffs({(1,0):1, (1,1):1}))
        
        for inner_t in range(7):
            # Inner Strassen operates on the 2x2 sub-matrices.
            # For a given outer term, we have a 4x4 coefficient matrix for A (U_outer_coeffs[outer_t]).
            # This 4x4 matrix represents linear combinations of the 2x2 blocks.
            # However, the inner Strassen acts on the 2x2 blocks themselves.
            # So we need to decompose the 2x2 block multiplication.
            
            # The structure is:
            # Term = sum_{i,j} U_outer[i,j] * A[i,j] * (sum_{k,l} V_outer[k,l] * B[k,l])
            # This is not quite right. The Strassen formula is:
            # M_outer = (Sum A_blocks) * (Sum B_blocks)
            # Each "Sum A_blocks" is a 2x2 matrix of 2x2 blocks? No.
            # The blocks A00, etc. are 2x2 matrices.
            # So (A00+A11) is a 2x2 matrix (sum of two 2x2 matrices).
            # So the product (A00+A11)(B00+B11) is a 2x2 matrix multiplication of two 2x2 matrices.
            # We apply Strassen to this 2x2 multiplication.
            
            # So, for outer term t, we define:
            # A_t = sum_{(br,bc)} coeff_A_t[br,bc] * A_block[br,bc]  (This is a 2x2 matrix)
            # B_t = sum_{(br,bc)} coeff_B_t[br,bc] * B_block[br,bc]  (This is a 2x2 matrix)
            # Then M_t = A_t * B_t (2x2 matrix multiplication)
            # We decompose M_t using inner Strassen.
            
            # The inner Strassen terms m1..m7 are linear combinations of elements of A_t and B_t.
            # A_t is 2x2. Elements A_t[0,0], A_t[0,1], etc.
            # A_t[0,0] = sum_{(br,bc)} coeff_A_t[br,bc] * A_block[br,bc][0,0]
            
            # We need to map the inner Strassen coefficients to the global 4x4 indices.
            
            # Inner U coeffs (for 2x2 matrix A_t)
            # u_inner[inner_t] is dict (r, c) -> coeff for A_t[r,c]
            # u_inner[0] = {(0,0):1, (1,1):1}
            # ... (same as U_coeffs defined earlier)
            
            # For each element (r,c) in A_t, it is a linear combination of A_block[br,bc][r,c].
            # So the total coefficient for A[2*br+r, 2*bc+c] in this term is:
            # coeff = u_inner[inner_t][(r,c)] * coeff_A_t[br,bc]
            
            # Similarly for B.
            
            # For W:
            # The result M_t is a 2x2 matrix of 2x2 blocks? No.
            # M_t is a 2x2 matrix.
            # M_t[r,c] contributes to C_block[br,bc][r,c]?
            # No, the outer Strassen reconstructs C blocks.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # etc.
            # So M_t contributes to specific C blocks.
            # And within that block, the inner Strassen reconstructs the 2x2 values.
            
            # W_coeffs_2x2[c_r, c_c] gives coeffs for M1..M7 (inner terms) for the 2x2 result element.
            # Wait, the inner Strassen produces the 2x2 matrix M_t.
            # So M_t[0,0] = m1 + m4 - m5 + m7 (using inner m's)
            # M_t[0,1] = m3 + m5
            # etc.
            
            # So for a specific global C index (I, K):
            # I = 2*br + r, K = 2*bc + c
            # The coefficient is sum_{inner_t} W_2x2[r,c][inner_t] * (Outer_W_coeff[outer_t] for this block)
            
            # Outer W coeffs:
            # Which outer terms contribute to which C blocks?
            # C00: M1(+1), M4(+1), M5(-1), M7(+1)
            # C01: M3(+1), M5(+1)
            # C10: M2(+1), M4(+1)
            # C11: M1(+1), M2(-1), M3(+1), M6(+1)
            
            # Let's build the full U, V, W columns.
            
            # 1. Fill U
            # For each (br, bc) in 0..1, 0..1:
            #   For each (r, c) in 0..1, 0..1:
            #     Global A index: i = 2*br + r, j = 2*bc + c
            #     Coeff from outer: U_outer_coeffs[outer_t][i, j]
            #     Coeff from inner: U_coeffs[inner_t][(r, c)]
            #     Total U[i, j, col] += coeff_outer * coeff_inner
            
            # 2. Fill V
            # For each (br, bc) in 0..1, 0..1:
            #   For each (r, c) in 0..1, 0..1:
            #     Global B index: k = 2*br + r, l = 2*bc + c
            #     Coeff from outer: V_outer_coeffs[outer_t][k, l]
            #     Coeff from inner: V_coeffs[inner_t][(r, c)]
            #     Total V[k, l, col] += coeff_outer * coeff_inner
            
            # 3. Fill W
            # For each C block (br, bc) in 0..1, 0..1:
            #   For each (r, c) in 0..1, 0..1:
            #     Global C index: i = 2*br + r, k = 2*bc + c
            #     Note: W index is k*n + i = k*4 + i.
            #     Coeff from outer: W_outer_coeffs[br, bc][outer_t]
            #     Coeff from inner: W_inner_coeffs[r, c][inner_t]
            #     Total W[k*4 + i, col] += coeff_outer * coeff_inner
            
            # Pre-compute outer W coeffs
            # W_outer_coeffs[br, bc] is a list of 7 coeffs for M1..M7
            W_outer_coeffs = np.zeros((2, 2, 7), dtype=int)
            # C00 (0,0): M1(+1), M4(+1), M5(-1), M7(+1) -> indices 0,3,4,6
            W_outer_coeffs[0,0,0] = 1
            W_outer_coeffs[0,0,3] = 1
            W_outer_coeffs[0,0,4] = -1
            W_outer_coeffs[0,0,6] = 1
            
            # C01 (0,1): M3(+1), M5(+1) -> indices 2,4
            W_outer_coeffs[0,1,2] = 1
            W_outer_coeffs[0,1,4] = 1
            
            # C10 (1,0): M2(+1), M4(+1) -> indices 1,3
            W_outer_coeffs[1,0,1] = 1
            W_outer_coeffs[1,0,3] = 1
            
            # C11 (1,1): M1(+1), M2(-1), M3(+1), M6(+1) -> indices 0,1,2,5
            W_outer_coeffs[1,1,0] = 1
            W_outer_coeffs[1,1,1] = -1
            W_outer_coeffs[1,1,2] = 1
            W_outer_coeffs[1,1,5] = 1

            # Pre-compute inner W coeffs
            # W_inner_coeffs[r, c] is a list of 7 coeffs for m1..m7
            W_inner_coeffs = np.zeros((2, 2, 7), dtype=int)
            # C00 (0,0): m1(+1), m4(+1), m5(-1), m7(+1)
            W_inner_coeffs[0,0,0] = 1
            W_inner_coeffs[0,0,3] = 1
            W_inner_coeffs[0,0,4] = -1
            W_inner_coeffs[0,0,6] = 1
            
            # C01 (0,1): m3(+1), m5(+1)
            W_inner_coeffs[0,1,2] = 1
            W_inner_coeffs[0,1,4] = 1
            
            # C10 (1,0): m2(+1), m4(+1)
            W_inner_coeffs[1,0,1] = 1
            W_inner_coeffs[1,0,3] = 1
            
            # C11 (1,1): m1(+1), m2(-1), m3(+1), m6(+1)
            W_inner_coeffs[1,1,0] = 1
            W_inner_coeffs[1,1,1] = -1
            W_inner_coeffs[1,1,2] = 1
            W_inner_coeffs[1,1,5] = 1

            # Now fill the column col
            # U
            for br in range(2):
                for bc in range(2):
                    for r in range(2):
                        for c in range(2):
                            i = 2*br + r
                            j = 2*bc + c
                            coeff_outer = U_outer_coeffs[outer_t][i, j]
                            if coeff_outer == 0: continue
                            coeff_inner = U_coeffs[inner_t].get((r,c), 0)
                            if coeff_inner == 0: continue
                            U[i, j, col] += coeff_outer * coeff_inner
                            
            # V
            for br in range(2):
                for bc in range(2):
                    for r in range(2):
                        for c in range(2):
                            k = 2*br + r
                            l = 2*bc + c
                            coeff_outer = V_outer_coeffs[outer_t][k, l]
                            if coeff_outer == 0: continue
                            coeff_inner = V_coeffs[inner_t].get((r,c), 0)
                            if coeff_inner == 0: continue
                            V[k, l, col] += coeff_outer * coeff_inner

            # W
            for br in range(2):
                for bc in range(2):
                    for r in range(2):
                        for c in range(2):
                            i = 2*br + r
                            k = 2*bc + c
                            w_idx = k * 4 + i
                            coeff_outer = W_outer_coeffs[br, bc, outer_t]
                            if coeff_outer == 0: continue
                            coeff_inner = W_inner_coeffs[r, c, inner_t]
                            if coeff_inner == 0: continue
                            W[w_idx, col] += coeff_outer * coeff_inner

            col += 1
            
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

