"""Rank-minimal exact tensor decomposition for 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _generate_strassen_4x4():
    """
    Generates the U, V, W matrices for the recursive Strassen decomposition of 4x4 matrices.
    Rank is 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize sparse representation: list of (row, col, value) for each matrix
    U_sparse = []
    V_sparse = []
    W_sparse = []

    # Strassen's 7 products for 2x2 matrices
    # Each product is defined by linear combinations of A and B elements
    # A elements are indexed (a_br, a_bc, a_coeff) where br, bc in {0,1}
    # B elements are indexed (b_br, b_bc, b_coeff)
    # The product result contributes to C elements via C_combination
    
    # Strassen 2x2 definitions:
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

    strassen_products = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: list of (local_row, local_col, coeff)
        # B_coeffs: list of (local_row, local_col, coeff)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),           # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),          # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),          # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # M7
    ]

    # C combination: C_local_row, C_local_col, list of (product_index, coeff)
    c_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)])
    ]

    # We have 4 block positions for the 4x4 matrices: (0,0), (0,1), (1,0), (1,1)
    # Each block is 2x2.
    # Global index for A: i = 2*block_row + local_row, j = 2*block_col + local_col
    # Global index for V: j = 2*block_row + local_row, k = 2*block_col + local_col
    # Global index for W: k = 2*block_row + local_row, i = 2*block_col + local_col
    
    # The recursive structure:
    # We perform 7 multiplications for each of the 4 block positions? 
    # No, Strassen applies to the whole 4x4 by splitting into 2x2 blocks.
    # So we have 4 block positions in A and B.
    # Wait, the standard Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The 7 products are formed from linear combinations of the 4x4 blocks.
    # Let A_blocks be A[br][bc] for br, bc in {0,1}.
    # Let B_blocks be B[br][bc] for br, bc in {0,1}.
    
    # The 7 products M1..M7 are products of linear combinations of A_blocks and B_blocks.
    # Each M_k is a 2x2 matrix product.
    # M_k = U_k * V_k where U_k is a 2x2 matrix of linear combos of A_blocks?
    # No, M_k = (Linear Combo of A blocks) * (Linear Combo of B blocks).
    # The result is a 2x2 matrix.
    # Then C_blocks are linear combos of M1..M7.
    
    # So, for each of the 7 Strassen products s (0..6):
    #   We form a product of two 2x2 matrices.
    #   The "left" matrix is a linear combination of A blocks.
    #   The "right" matrix is a linear combination of B blocks.
    #   This product contributes to the C blocks via the c_combination.
    
    # Let's iterate over the 7 Strassen products s.
    # For each s, we have a set of A-block coefficients and B-block coefficients.
    # But wait, the A and B inputs to Strassen are the 4x4 matrices themselves.
    # The 7 products are:
    # M1 = (A00+A11) * (B00+B11)
    # Here A00 is the top-left 2x2 block of A.
    # So A00 corresponds to global rows 0-1, cols 0-1.
    # A11 corresponds to global rows 2-3, cols 2-3.
    
    # So for product s, we have a linear combination of A blocks.
    # Each A block is 2x2.
    # So the "U" part for product s is a 2x2 matrix? No.
    # The product M_s is a 2x2 matrix.
    # M_s[i,j] = sum_{k,l} U_s[i,k] V_s[k,j]? No.
    # M_s = A_s * B_s where A_s and B_s are 2x2 matrices.
    # A_s is a linear combination of A_blocks.
    # B_s is a linear combination of B_blocks.
    
    # Let's refine:
    # A_s = sum_{br, bc} coeff_A_s[br][bc] * A_block[br][bc]
    # B_s = sum_{br, bc} coeff_B_s[br][bc] * B_block[br][bc]
    
    # The tensor decomposition term for this product s is:
    # sum_{i,j,k in 0..1} A_s[i,j] B_s[j,k] ... wait, matrix mult is sum_k.
    # M_s[i,k] = sum_j A_s[i,j] B_s[j,k]
    
    # The global indices:
    # A_block[br][bc] has global indices i=2*br+li, j=2*bc+lj.
    # So A_s[li, lj] is not a single value, it's a sum over blocks.
    # Actually, A_s is a 2x2 matrix where each entry is a linear combination of 4x4 entries?
    # No, A_s is a 2x2 matrix of coefficients for the blocks.
    # Let's look at M1 = (A00+A11)(B00+B11).
    # A00 is 2x2. A11 is 2x2.
    # A_s = A00 + A11. This is a 2x2 matrix.
    # B_s = B00 + B11. This is a 2x2 matrix.
    # M1 = A_s * B_s.
    # M1 is 2x2.
    
    # The contribution to the final 4x4 C matrix:
    # C_block[0,0] += M1
    # C_block[0,1] += ...
    
    # So, for each s in 0..6:
    #   We define A_s as a 2x2 matrix of coefficients for the 4 A-blocks.
    #   We define B_s as a 2x2 matrix of coefficients for the 4 B-blocks.
    #   The product M_s = A_s * B_s is a 2x2 matrix.
    #   Each entry M_s[li, lk] contributes to C_block[cbr, cbc] via c_combination.
    
    # Let's map this to the rank-49 decomposition.
    # The rank is 49 because each of the 7 products M_s (which are 2x2) has rank 7 (Strassen rank for 2x2).
    # So M_s = sum_{r=0}^6 u_{s,r} v_{s,r} w_{s,r} (tensor rank 7).
    # Total rank = 7 * 7 = 49.
    
    # We need to expand each M_s into its 7 rank-1 terms.
    
    # Strassen 2x2 decomposition (rank 7):
    # For a 2x2 multiplication, the 7 terms are defined by the strassen_products list above.
    # Let's call the 7 terms of the 2x2 multiplication t0..t6.
    # For each t, we have A_t_coeffs and B_t_coeffs (for the 2x2 inputs).
    
    # So, for the 4x4 case:
    # For each s in 0..6 (the 7 Strassen products of the 4x4 blocks):
    #   A_s is a 2x2 matrix of 2x2 blocks.
    #   B_s is a 2x2 matrix of 2x2 blocks.
    #   M_s = A_s * B_s.
    #   We decompose M_s using the 2x2 Strassen decomposition.
    #   The inputs to the 2x2 Strassen decomposition are the 2x2 matrices A_s and B_s.
    #   But A_s and B_s are themselves linear combinations of 4x4 blocks.
    
    # Let's trace the indices carefully.
    # Global U matrix: 16x49.
    # Row index: i*4 + j.
    # Col index: s*7 + t, where s in 0..6, t in 0..6.
    
    # For a fixed s and t:
    # The term is from the t-th Strassen product of the 2x2 multiplication M_s.
    # The t-th product is A_t * B_t, where A_t and B_t are linear combinations of the 2x2 entries of A_s and B_s.
    
    # A_t = sum_{li, lj} coeff_A_t[li, lj] * A_s[li, lj]
    # B_t = sum_{lj, lk} coeff_B_t[lj, lk] * B_s[lj, lk]
    
    # A_s[li, lj] = sum_{br, bc} coeff_A_s[li, lj, br, bc] * A_block[br, bc][li, lj]
    # Wait, A_s is a 2x2 matrix. Its entry (li, lj) is a linear combination of the 4 A-blocks' (li, lj) entries.
    # A_s[li, lj] = sum_{br, bc} alpha_s[li, lj, br, bc] * A[2*br+li, 2*bc+lj]
    
    # So A_t[li, lj] = sum_{br, bc} (sum_{li', lj'} coeff_A_t[li', lj'] * alpha_s[li', lj', br, bc]) * A[2*br+li, 2*bc+lj]
    # This is getting complicated. Let's just build the matrices directly.
    
    # Step 1: Define the 4x4 Strassen structure.
    # The 7 products M_s are defined by linear combinations of the 4x4 blocks.
    # M_s = (sum_{br,bc} alpha_s[br,bc] A_block[br,bc]) * (sum_{br,bc} beta_s[br,bc] B_block[br,bc])
    
    # alpha_s and beta_s are 2x2 matrices of coefficients.
    # From the standard Strassen formulas for 4x4:
    # M1 = (A00+A11)(B00+B11) -> alpha1 = [[1,0],[0,1]], beta1 = [[1,0],[0,1]]
    # M2 = (A10+A11)B00 -> alpha2 = [[0,0],[1,1]], beta2 = [[1,0],[0,0]]
    # M3 = A00(B01-B11) -> alpha3 = [[1,0],[0,0]], beta3 = [[0,1],[0,-1]]
    # M4 = A11(B10-B00) -> alpha4 = [[0,0],[0,1]], beta4 = [[0,-1],[1,0]]
    # M5 = (A00+A01)B11 -> alpha5 = [[1,1],[0,0]], beta5 = [[0,0],[0,1]]
    # M6 = (A10-A00)(B00+B01) -> alpha6 = [[-1,0],[1,0]], beta6 = [[1,1],[0,0]]
    # M7 = (A01-A11)(B10+B11) -> alpha7 = [[0,1],[-1,0]], beta7 = [[0,1],[1,1]]
    
    # Note: A_block[br,bc] is the 2x2 submatrix at block (br,bc).
    # So A_s[li, lj] = sum_{br,bc} alpha_s[br,bc] * A[2*br+li, 2*bc+lj]
    # Similarly for B_s.
    
    # Step 2: For each s, decompose the 2x2 multiplication M_s = A_s * B_s using Strassen.
    # The 2x2 Strassen products t=0..6 are defined by:
    # A_t = sum_{li,lj} coeff_A_2x2_t[li,lj] A_s[li,lj]
    # B_t = sum_{lj,lk} coeff_B_2x2_t[lj,lk] B_s[lj,lk]
    
    # Step 3: The result M_s[li,lk] = sum_{lj} A_s[li,lj] B_s[lj,lk]
    # This M_s contributes to C_block[cbr,cbc] via c_combination.
    # C_block[cbr,cbc][li,lk] += sum_s gamma_s[cbr,cbc] * M_s[li,lk]
    
    # Combining all:
    # C[2*cbr+li, 2*cbc+lk] = sum_s sum_t gamma_s[cbr,cbc] * (A_t * B_t)[li,lk]
    # (A_t * B_t)[li,lk] = sum_lj A_t[li,lj] B_t[lj,lk]
    
    # A_t[li,lj] = sum_{br,bc} alpha_s[br,bc] * coeff_A_2x2_t[li,lj] * A[2*br+li, 2*bc+lj]
    # B_t[lj,lk] = sum_{br,bc} beta_s[br,bc] * coeff_B_2x2_t[lj,lk] * B[2*br+lj, 2*bc+lk]
    
    # So the coefficient for U[2*br+li, 2*bc+lj, s*7+t] is:
    # U_coeff = alpha_s[br,bc] * coeff_A_2x2_t[li,lj]
    
    # The coefficient for V[2*br+lj, 2*bc+lk, s*7+t] is:
    # V_coeff = beta_s[br,bc] * coeff_B_2x2_t[lj,lk]
    
    # The coefficient for W[2*cbc+lk, 2*cbr+li, s*7+t] is:
    # W_coeff = gamma_s[cbr,cbc]
    
    # Wait, the W matrix maps the product index to the output.
    # The product index is (s, t, li, lk)? No, the rank index is r = s*7+t.
    # The term in the sum is U[i,j,r] V[j,k,r] W[k,i,r].
    # Here i,j are A indices, j,k are B indices, k,i are C indices.
    # My derivation:
    # C[2*cbr+li, 2*cbc+lk] gets contribution from r=s*7+t.
    # The contribution is gamma_s[cbr,cbc] * (A_t * B_t)[li,lk].
    # (A_t * B_t)[li,lk] = sum_lj A_t[li,lj] B_t[lj,lk].
    # A_t[li,lj] is a linear combo of A entries.
    # B_t[lj,lk] is a linear combo of B entries.
    # So U[i,j,r] should be the coefficient of A[i,j] in A_t[li,lj].
    # i = 2*br+li, j = 2*bc+lj.
    # U[i,j,r] = alpha_s[br,bc] * coeff_A_2x2_t[li,lj]
    
    # V[j,k,r] should be the coefficient of B[j,k] in B_t[lj,lk].
    # j = 2*br+lj, k = 2*bc+lk.
    # V[j,k,r] = beta_s[br,bc] * coeff_B_2x2_t[lj,lk]
    
    # W[k,i,r] should be the coefficient of the product term in C[k,i].
    # Here the product term is (A_t*B_t)[li,lk] which contributes to C[2*cbr+li, 2*cbc+lk].
    # So k = 2*cbc+lk, i = 2*cbr+li.
    # W[k,i,r] = gamma_s[cbr,cbc]
    
    # This looks consistent.
    
    # Let's implement this.
    
    alpha_s = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]]
    ]
    
    beta_s = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[0,-1],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,1],[1,1]]
    ]
    
    # gamma_s[cbr,cbc] is the coefficient of M_s in C_block[cbr,cbc].
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    gamma_s = [
        # s=0 (M1)
        [[1, 0], [0, 1]], # C00=1, C01=0, C10=0, C11=1
        # s=1 (M2)
        [[0, 0], [1, -1]], # C00=0, C01=0, C10=1, C11=-1
        # s=2 (M3)
        [[0, 1], [0, 1]], # C00=0, C01=1, C10=0, C11=1
        # s=3 (M4)
        [[1, 0], [1, 0]], # C00=1, C01=0, C10=1, C11=0
        # s=4 (M5)
        [[-1, 1], [0, 0]], # C00=-1, C01=1, C10=0, C11=0
        # s=5 (M6)
        [[0, 0], [0, 1]], # C00=0, C01=0, C10=0, C11=1
        # s=6 (M7)
        [[1, 0], [0, 0]]  # C00=1, C01=0, C10=0, C11=0
    ]
    
    # 2x2 Strassen coefficients (coeff_A_2x2_t, coeff_B_2x2_t)
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # coeff_A_2x2_t is 2x2 matrix of coefficients for A_s entries.
    # A_t[li,lj] = sum coeff_A_2x2_t[li,lj] A_s[li,lj]
    # Wait, A_t is a 2x2 matrix.
    # A_t[0,0] = A_s[0,0] + A_s[1,1]
    # A_t[0,1] = 0
    # A_t[1,0] = 0
    # A_t[1,1] = 0
    # So coeff_A_2x2_0 = [[1,0],[0,1]]? No.
    # A_t is the matrix multiplied.
    # A_t = [[A00+A11, 0], [0, 0]]? No.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # Let P1 = A00+A11. This is a scalar? No, A00 is a block?
    # In the 2x2 case, A00 is a scalar.
    # So P1 = A00+A11.
    # Q1 = B00+B11.
    # M1 = P1 * Q1.
    # So A_t is a 1x1 matrix? No, the product is a scalar.
    # The 2x2 multiplication is decomposed into 7 scalar products.
    # So A_t and B_t are scalars (linear combos of the 4 entries).
    
    # So coeff_A_2x2_t is a vector of 4 coefficients for A[0,0], A[0,1], A[1,0], A[1,1].
    # A_t = c00*A00 + c01*A01 + c10*A10 + c11*A11
    
    coeff_A_2x2 = [
        [1, 0, 0, 1], # t=0: A00+A11
        [0, 0, 1, 1], # t=1: A10+A11
        [1, 0, 0, 0], # t=2: A00
        [0, 0, 0, 1], # t=3: A11
        [1, 1, 0, 0], # t=4: A00+A01
        [-1, 0, 1, 0],# t=5: A10-A00
        [0, 1, 0, -1] # t=6: A01-A11
    ]
    
    coeff_B_2x2 = [
        [1, 0, 0, 1], # t=0: B00+B11
        [1, 0, 0, 0], # t=1: B00
        [0, 1, 0, -1],# t=2: B01-B11
        [0, -1, 1, 0],# t=3: B10-B00
        [0, 0, 0, 1], # t=4: B11
        [1, 1, 0, 0], # t=5: B00+B01
        [0, 1, 1, 1]  # t=6: B10+B11
    ]
    
    # Now we can build U, V, W.
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # A_t = sum_{li,lj} coeff_A_2x2[t][li*2+lj] * A_s[li,lj]
            # A_s[li,lj] = sum_{br,bc} alpha_s[br,bc] * A[2*br+li, 2*bc+lj]
            # So A_t = sum_{li,lj,br,bc} coeff_A_2x2[t][li*2+lj] * alpha_s[br,bc] * A[2*br+li, 2*bc+lj]
            
            # U[i,j,r] = coeff_A_2x2[t][li*2+lj] * alpha_s[br,bc]
            # where i = 2*br+li, j = 2*bc+lj
            
            for li in range(2):
                for lj in range(2):
                    c_a = coeff_A_2x2[t][li*2+lj]
                    if c_a == 0: continue
                    for br in range(2):
                        for bc in range(2):
                            a_s_coeff = alpha_s[s][br][bc]
                            if a_s_coeff == 0: continue
                            u_coeff = c_a * a_s_coeff
                            i = 2*br + li
                            j = 2*bc + lj
                            U[i*4 + j, r] = u_coeff # Should be +=, but each (i,j,r) is unique?
                            # i,j are determined by br,bc,li,lj. r is s,t.
                            # Yes, unique.
                            
            # B_t = sum_{lj,lk} coeff_B_2x2[t][lj*2+lk] * B_s[lj,lk]
            # B_s[lj,lk] = sum_{br,bc} beta_s[br,bc] * B[2*br+lj, 2*bc+lk]
            
            for lj in range(2):
                for lk in range(2):
                    c_b = coeff_B_2x2[t][lj*2+lk]
                    if c_b == 0: continue
                    for br in range(2):
                        for bc in range(2):
                            b_s_coeff = beta_s[s][br][bc]
                            if b_s_coeff == 0: continue
                            v_coeff = c_b * b_s_coeff
                            j = 2*br + lj
                            k = 2*bc + lk
                            V[j*4 + k, r] = v_coeff
                            
            # W[k,i,r] = gamma_s[cbr,cbc]
            # The product M_s[li,lk] contributes to C[2*cbr+li, 2*cbc+lk]
            # So k = 2*cbc+lk, i = 2*cbr+li
            # W[k*4 + i, r] = gamma_s[s][cbr][cbc]
            
            for cbr in range(2):
                for cbc in range(2):
                    g = gamma_s[s][cbr][cbc]
                    if g == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            k = 2*cbc + lk
                            i = 2*cbr + li
                            W[k*4 + i, r] = g
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute the decomposition
_DECOMPOSITION = _generate_strassen_4x4()

def construct():
    """
    Returns the U, V, W matrices for the rank-49 decomposition.
    """
    return _DECOMPOSITION

# EVOLVE-BLOCK-END
