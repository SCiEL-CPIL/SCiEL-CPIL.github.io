import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices representing the tensor decomposition.
    Uses a hybrid decomposition:
    - Split B (4x6) into B1 (4x4) and B2 (4x2).
    - Compute A*B1 using Strassen's algorithm (Rank 49).
    - Compute A*B2 using schoolbook multiplication (Rank 32).
    Total Rank: 81.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS # 4, 4, 6
    rank_total = 81
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col_idx
        if col_idx >= rank_total:
            return
        U[u_idx][col_idx] += val
        V[v_idx][col_idx] += val
        W[w_idx][col_idx] += val
        col_idx += 1

    # Part 1: Strassen 4x4x4
    # We need to decompose the 4x4x4 tensor into 49 rank-1 terms.
    # This is done by applying Strassen to the 2x2 blocks, where each block multiplication is also Strassen.
    
    # Define the 7 Strassen terms for the block level
    # Each term is defined by:
    # - A coefficients for the 4 blocks (A00, A01, A10, A11)
    # - B coefficients for the 4 blocks (B00, B01, B10, B11)
    # - C coefficients for the 4 blocks (C00, C01, C10, C11) with signs
    # Note: The standard Strassen decomposition gives C as linear combination of Ms.
    # Here we are expanding M_k = (Sum a_i A_i) (Sum b_j B_j).
    # Then C_l = Sum_k coeff_lk M_k.
    # So for each k, we have a product of two linear combos.
    # We need to expand (Sum a_i A_i) (Sum b_j B_j) into rank-1 terms?
    # No, A_i and B_j are matrices. The product of two matrices is not rank 1.
    # We must use the Strassen decomposition for the 2x2 matrix multiplication inside each M_k.
    
    # So, for each of the 7 block-level terms M_k:
    # 1. Identify the linear combination of A-blocks: A_comb = Sum_i alpha_i A_i
    # 2. Identify the linear combination of B-blocks: B_comb = Sum_j beta_j B_j
    # 3. The product M_k = A_comb * B_comb.
    # 4. We need to decompose A_comb * B_comb.
    #    A_comb is a 2x2 matrix. B_comb is a 2x2 matrix.
    #    We can use the standard Strassen decomposition for 2x2 matrices to decompose this product into 7 rank-1 terms.
    #    However, the entries of A_comb and B_comb are linear combinations of the entries of A and B.
    #    This means each of the 7 rank-1 terms for the 2x2 product will involve linear combinations of the original A and B entries.
    
    # Let's define the 2x2 Strassen decomposition first.
    # For X(2x2) * Y(2x2) = Z(2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M6 + M7
    
    # We will iterate through the 7 block-level terms.
    # For each term, we have a set of (alpha_i, beta_j) pairs that define A_comb and B_comb.
    # Actually, it's easier to just list the 7 block terms and their contributions to C blocks.
    
    # Block Term Definitions:
    # Term 1: A_comb = A00 + A11, B_comb = B00 + B11
    # Term 2: A_comb = A10 + A11, B_comb = B00
    # Term 3: A_comb = A00, B_comb = B01 - B11
    # Term 4: A_comb = A11, B_comb = B10 - B00
    # Term 5: A_comb = A00 + A01, B_comb = B11
    # Term 6: A_comb = A10 - A00, B_comb = B00 + B01
    # Term 7: A_comb = A01 - A11, B_comb = B10 + B11
    
    # C Block Contributions:
    # C00: +M1, +M4, -M5, +M7
    # C01: +M3, +M5
    # C10: +M2, +M4
    # C11: +M1, -M2, +M6, +M7
    
    # We will generate the 49 terms by iterating over the 7 block terms and then the 7 sub-terms of the 2x2 Strassen.
    
    # Helper to map block indices to global indices
    # A block (bi, bj) where bi, bj in {0,1}
    # Global A index for element (r, c) in block (bi, bj) is (bi*2+r)*4 + (bj*2+c)
    # B block (bi, bj)
    # Global B index for element (r, c) in block (bi, bj) is (bi*2+r)*6 + (bj*2+c)  (Note: B has 6 cols, but we only use 0-3 here)
    # C block (ci, cj)
    # Global C index for element (r, c) in block (ci, cj) is (ci*2+r)*4 + (cj*2+c)
    # W index for C element (i, k) is k*4 + i.
    # Here C element (r+c_i*2, c+c_j*2) -> i = r+c_i*2, k = c+c_j*2.
    # W index = (c+c_j*2)*4 + (r+c_i*2).
    
    # Define the 7 block terms
    block_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict { (bi,bj): coeff }
        # B_coeffs: dict { (bi,bj): coeff }
        # C_coeffs: dict { (ci,cj): coeff }
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # M2 -> C10
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # M3 -> C01
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4 -> C00, C10
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5 -> C00, C01
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6 -> C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M7 -> C00, C11
    ]
    
    # Define the 2x2 Strassen sub-terms
    # For X*Y=Z:
    # Sub-term 1: X00+X11, Y00+Y11 -> Z00+1, Z11+1
    # Sub-term 2: X10+X11, Y00 -> Z10+1
    # Sub-term 3: X00, Y01-Y11 -> Z01+1
    # Sub-term 4: X11, Y10-Y00 -> Z00+1, Z10+1
    # Sub-term 5: X00+X01, Y11 -> Z00-1, Z01+1
    # Sub-term 6: X10-X00, Y00+Y01 -> Z11+1
    # Sub-term 7: X01-X11, Y10+Y11 -> Z00+1, Z11+1
    
    sub_terms = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # X_coeffs: dict {(r,c): coeff}
        # Y_coeffs: dict {(r,c): coeff}
        # Z_coeffs: dict {(r,c): coeff}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    for b_idx, (a_block_coeffs, b_block_coeffs, c_block_coeffs) in enumerate(block_terms):
        for s_idx, (x_coeffs, y_coeffs, z_coeffs) in enumerate(sub_terms):
            # We need to form the rank-1 term.
            # The term is (Sum_A) * (Sum_B) * (Sum_C)
            # Sum_A is a linear combination of A entries.
            # Sum_B is a linear combination of B entries.
            # Sum_C is a linear combination of C entries (which maps to W).
            
            # We can't just add a single rank-1 term here because Sum_A and Sum_B are sums of multiple entries.
            # We must expand the product of sums into individual rank-1 terms?
            # No, the tensor decomposition allows coefficients in U, V, W.
            # A rank-1 term is u * v * w.
            # If u = u1 + u2, then (u1+u2)v w = u1 v w + u2 v w.
            # So we need to split the linear combinations into individual basis vectors.
            
            # Let's collect the coefficients for U, V, W for this specific sub-term.
            # For each element in X_coeffs (which corresponds to A entries):
            # For each element in Y_coeffs (which corresponds to B entries):
            # For each element in Z_coeffs (which corresponds to C entries):
            # This would create a term for every combination.
            # But wait, the Strassen decomposition works by defining M_k as a product of two *matrices*.
            # The matrices X and Y are linear combinations of the input matrices.
            # The product X*Y is decomposed into 7 rank-1 terms in the space of *matrix entries*?
            # No, the rank-1 terms in the tensor decomposition correspond to scalar multiplications.
            # The standard Strassen decomposition for 2x2 matrices has rank 7.
            # This means there are 7 scalar multiplications.
            # Each scalar multiplication involves a linear combination of X entries and a linear combination of Y entries.
            # The result is added to a linear combination of Z entries.
            
            # So, for each sub-term (one of the 7 in 2x2 Strassen):
            # We have:
            # X_part = Sum_{(r,c) in X_coeffs} x_val * X_{r,c}
            # Y_part = Sum_{(r,c) in Y_coeffs} y_val * Y_{r,c}
            # Z_part = Sum_{(r,c) in Z_coeffs} z_val * Z_{r,c}
            # The equation is X_part * Y_part = Z_part.
            # This is NOT a single rank-1 term in the tensor decomposition of the whole matrix multiplication.
            # It is a rank-1 term in the *block* multiplication? No.
            
            # Let's re-evaluate.
            # The tensor T is defined by T[i,j,k] = delta_{i,j} delta_{j,k} (for schoolbook).
            # A rank-1 decomposition is T = Sum_r u_r v_r w_r.
            # u_r is a vector of length n*m.
            # v_r is a vector of length m*p.
            # w_r is a vector of length p*n.
            
            # When we use Strassen, we are effectively changing the basis.
            # The 7 terms M_k are scalar products of linear combinations.
            # M_k = (Sum_i a_i A_i) * (Sum_j b_j B_j).
            # This M_k is a scalar? No, M_k is a matrix.
            # In the 2x2 case, M_k is a 2x2 matrix? No, M_k is a scalar product of two 2x2 matrices?
            # No, in Strassen 2x2, X and Y are 2x2 matrices.
            # M1 = (X00+X11)(Y00+Y11). This is a scalar product of two scalars.
            # So M1 is a scalar.
            # Then Z00 = M1 + M4 - M5 + M7.
            # So Z is a matrix of linear combinations of these 7 scalars.
            
            # So, for the 2x2 Strassen decomposition:
            # We have 7 scalar products.
            # Each scalar product is (Linear Combo of X entries) * (Linear Combo of Y entries).
            # And it contributes to (Linear Combo of Z entries).
            
            # So, for each of the 7 sub-terms:
            # We can define:
            # u_vec: vector over A entries.
            # v_vec: vector over B entries.
            # w_vec: vector over C entries.
            # The term is u_vec * v_vec * w_vec.
            # This is a valid rank-1 term in the tensor decomposition.
            
            # So, we just need to construct u_vec, v_vec, w_vec for each sub-term.
            
            # u_vec components:
            # For each (br, bc) in a_block_coeffs with coeff a_val:
            #   For each (sr, sc) in x_coeffs with coeff x_val:
            #     The A entry is at block (br, bc), sub-index (sr, sc).
            #     Global A index: (br*2+sr)*4 + (bc*2+sc).
            #     Coefficient: a_val * x_val.
            
            # v_vec components:
            # For each (br, bc) in b_block_coeffs with coeff b_val:
            #   For each (sr, sc) in y_coeffs with coeff y_val:
            #     The B entry is at block (br, bc), sub-index (sr, sc).
            #     Global B index: (br*2+sr)*6 + (bc*2+sc).
            #     Coefficient: b_val * y_val.
            
            # w_vec components:
            # For each (cr, cc) in c_block_coeffs with coeff c_val:
            #   For each (sr, sc) in z_coeffs with coeff z_val:
            #     The C entry is at block (cr, cc), sub-index (sr, sc).
            #     Global C index: i = cr*2+sr, k = cc*2+sc.
            #     W index: k*4 + i.
            #     Coefficient: c_val * z_val.
            
            # We can add this as a single rank-1 term by populating U, V, W with these coefficients.
            
            for (br, bc), a_val in a_block_coeffs.items():
                for (sr, sc), x_val in x_coeffs.items():
                    a_idx = (br*2+sr)*4 + (bc*2+sc)
                    U[a_idx][col_idx] += a_val * x_val
            
            for (br, bc), b_val in b_block_coeffs.items():
                for (sr, sc), y_val in y_coeffs.items():
                    b_idx = (br*2+sr)*6 + (bc*2+sc)
                    V[b_idx][col_idx] += b_val * y_val
            
            for (cr, cc), c_val in c_block_coeffs.items():
                for (sr, sc), z_val in z_coeffs.items():
                    c_i = cr*2+sr
                    c_k = cc*2+sc
                    w_idx = c_k*4 + c_i
                    W[w_idx][col_idx] += c_val * z_val
            
            col_idx += 1
    
    # Part 2: Schoolbook 4x4x2
    # B columns 4, 5.
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_idx = i*4 + j
                v_idx = j*6 + k
                w_idx = k*4 + i
                U[u_idx][col_idx] = 1
                V[v_idx][col_idx] = 1
                W[w_idx][col_idx] = 1
                col_idx += 1
                
    if time.time() > deadline:
        # Return best effort
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
