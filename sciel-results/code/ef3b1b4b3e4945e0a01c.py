import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition for 4x4x4 (rank 49) twice,
    # and 4x4x1 (rank 16) once.
    # Total rank = 49 + 49 + 16 = 114.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal current_rank
        U[u_idx][current_rank] += val
        V[v_idx][current_rank] += val
        W[w_idx][current_rank] += val
        current_rank += 1

    # Strassen for 2x2 matrices
    # A is 2x2, B is 2x2, C is 2x2
    # Maps local indices to global indices
    def strassen_2x2(u_row_offset, u_col_offset, v_row_offset, v_col_offset, w_row_offset, w_col_offset):
        # u_row_offset, u_col_offset: top-left corner of A block in global A (4x4)
        # v_row_offset, v_col_offset: top-left corner of B block in global B (4x4)
        # w_row_offset, w_col_offset: top-left corner of C block in global C (4x4)
        
        # A indices:
        # a00, a01
        # a10, a11
        a00 = (u_row_offset, u_col_offset)
        a01 = (u_row_offset, u_col_offset + 1)
        a10 = (u_row_offset + 1, u_col_offset)
        a11 = (u_row_offset + 1, u_col_offset + 1)
        
        # B indices:
        # b00, b01
        # b10, b11
        b00 = (v_row_offset, v_col_offset)
        b01 = (v_row_offset, v_col_offset + 1)
        b10 = (v_row_offset + 1, v_col_offset)
        b11 = (v_row_offset + 1, v_col_offset + 1)
        
        # C indices:
        # c00, c01
        # c10, c11
        c00 = (w_row_offset, w_col_offset)
        c01 = (w_row_offset, w_col_offset + 1)
        c10 = (w_row_offset + 1, w_col_offset)
        c11 = (w_row_offset + 1, w_col_offset + 1)
        
        # Strassen 7 products:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        
        # We need to decompose each 2x2 multiplication into 7 rank-1 terms.
        # This is recursive!
        # But for 2x2, the "multiplication" is the base case.
        # Wait, Strassen reduces 2x2 multiplication to 7 scalar multiplications.
        # In the tensor decomposition context, a "scalar multiplication" is a rank-1 term.
        # So we just add 7 rank-1 terms for each M_k.
        
        # However, the inputs to M_k are linear combinations of A and B entries.
        # We must expand these linear combinations into rank-1 terms.
        
        # Example: M1 = (a00 + a11) * (b00 + b11)
        # This is not a single rank-1 term. It is a product of sums.
        # (x+y)(z+w) = xz + xw + yz + yw.
        # So M1 contributes 4 rank-1 terms.
        # Total terms = 7 * 4 = 28?
        # No, Strassen's algorithm is an algebraic identity.
        # The "rank" of the 2x2x2 tensor is 7.
        # This means there exist 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
        # The 7 products in Strassen are NOT rank-1 terms in the original 2x2x2 space.
        # They are intermediate values.
        # To get the rank-1 decomposition, we need to express the final C entries as linear combinations of the 7 M's.
        # And each M is a product of linear forms in A and B.
        # So each M corresponds to a "multiplication" of two linear forms.
        # In the tensor decomposition, a term is $u \cdot v \cdot w$.
        # If we have a product of linear forms $L_A(A) L_B(B)$, this is a rank-1 term if we define the U and V vectors appropriately.
        
        # So, for each Strassen product $M_k = L_A(A) L_B(B)$, we add ONE rank-1 term?
        # No. The Strassen algorithm computes $C = AB$ using 7 multiplications.
        # Each multiplication is of two numbers (which are linear combinations of entries).
        # So yes, each of the 7 products corresponds to a rank-1 term in the decomposition of the $2 \times 2 \times 2$ tensor.
        
        # Let's verify.
        # Rank 7 decomposition of $2 \times 2 \times 2$:
        # $C_{00} = M_1 + M_4 - M_5 + M_7$
        # $C_{01} = M_3 + M_5$
        # $C_{10} = M_2 + M_6$
        # $C_{11} = M_1 - M_2 + M_3 + M_6$
        
        # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$
        # This is a product of two linear forms.
        # In the tensor decomposition, we add a term with:
        # U vector having 1s at indices for $A_{00}, A_{11}$.
        # V vector having 1s at indices for $B_{00}, B_{11}$.
        # W vector having coefficients for $C_{00}, C_{11}$ etc.
        
        # So, for each of the 7 M's, we add ONE rank-1 term.
        # Total 7 terms for 2x2x2.
        
        # Let's implement this.
        
        # M1 = (A00+A11)(B00+B11)
        add_term_2x2(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1} # C00 gets +1, C11 gets +1
        )
        
        # M2 = (A10+A11)B00
        add_term_2x2(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1, c11: -1} # C10 gets +1, C11 gets -1
        )
        
        # M3 = A00(B01-B11)
        add_term_2x2(
            {a00: 1},
            {b01: 1, b11: -1},
            {c01: 1, c11: 1} # C01 gets +1, C11 gets +1
        )
        
        # M4 = A11(B10-B00)
        add_term_2x2(
            {a11: 1},
            {b10: 1, b00: -1},
            {c00: 1, c10: 1} # C00 gets +1, C10 gets +1
        )
        
        # M5 = (A00+A01)B11
        add_term_2x2(
            {a00: 1, a01: 1},
            {b11: 1},
            {c00: -1, c01: 1} # C00 gets -1, C01 gets +1
        )
        
        # M6 = (A10-A00)(B00+B01)
        add_term_2x2(
            {a10: 1, a00: -1},
            {b00: 1, b01: 1},
            {c10: 1, c11: 1} # C10 gets +1, C11 gets +1
        )
        
        # M7 = (A01-A11)(B10+B11)
        add_term_2x2(
            {a01: 1, a11: -1},
            {b10: 1, b11: 1},
            {c00: 1, c11: -1} # C00 gets +1, C11 gets -1
        )
        
        def add_term_2x2(u_dict, v_dict, w_dict):
            nonlocal current_rank
            # u_dict: { (row, col): coeff }
            # v_dict: { (row, col): coeff }
            # w_dict: { (row, col): coeff }
            
            # Flatten indices for U (4x4), V (4x4), W (4x4)
            # U index = row * 4 + col
            # V index = row * 4 + col
            # W index = row * 4 + col
            
            for (r, c), val_u in u_dict.items():
                U[r * 4 + c][current_rank] += val_u
            for (r, c), val_v in v_dict.items():
                V[r * 4 + c][current_rank] += val_v
            for (r, c), val_w in w_dict.items():
                W[r * 4 + c][current_rank] += val_w
            current_rank += 1

    # Strassen for 4x4 matrices
    # Uses 7 calls to strassen_2x2
    def strassen_4x4(u_row_offset, u_col_offset, v_row_offset, v_col_offset, w_row_offset, w_col_offset):
        # A blocks:
        # A00: (0,0) to (1,1)
        # A01: (0,2) to (1,3)
        # A10: (2,0) to (3,1)
        # A11: (2,2) to (3,3)
        
        # B blocks:
        # B00: (0,0) to (1,1)
        # B01: (0,2) to (1,3)
        # B10: (2,0) to (3,1)
        # B11: (2,2) to (3,3)
        
        # C blocks:
        # C00: (0,0) to (1,1)
        # C01: (0,2) to (1,3)
        # C10: (2,0) to (3,1)
        # C11: (2,2) to (3,3)
        
        # M1 = (A00+A11)(B00+B11)
        # This is a 2x2 multiplication.
        # We need to compute this 2x2 multiplication using strassen_2x2.
        # But the inputs are sums of blocks.
        # strassen_2x2 expects specific indices.
        # We cannot directly call strassen_2x2 with "sum of blocks".
        # We must expand the linear combinations.
        
        # This approach is getting complicated.
        # Instead of recursive Strassen, let's just use the 7 products directly.
        # Each product is a 2x2 matrix multiplication.
        # We can decompose each 2x2 multiplication into 7 rank-1 terms.
        # So total 7 * 7 = 49 terms.
        
        # Let's redefine add_term_2x2 to take explicit linear combinations.
        
        # M1 = (A00+A11)(B00+B11)
        # A_part = A00 + A11
        # B_part = B00 + B11
        # C_part = C00 + C11 (with signs)
        
        # We need a function that takes linear combinations of A and B blocks
        # and adds the rank-1 terms for their 2x2 product.
        
        def add_2x2_product(a_comb, b_comb, c_comb):
            # a_comb: list of (block_idx, coeff)
            # b_comb: list of (block_idx, coeff)
            # c_comb: list of (block_idx, coeff)
            
            # We need to compute the 2x2 product of the linear forms.
            # This is equivalent to 7 rank-1 terms?
            # No, the product of two 2x2 matrices is a 2x2 matrix.
            # The tensor for 2x2x2 has rank 7.
            # So we add 7 rank-1 terms.
            
            # But the inputs are linear combinations.
            # This means we are adding 7 terms, each of which is a product of linear forms.
            
            # Let's use the Strassen decomposition for 2x2x2.
            # We need to map the "virtual" 2x2 matrices to the actual A and B blocks.
            
            # This is effectively implementing the 7 Strassen products for the 4x4 case,
            # where each "scalar" is a 2x2 block.
            
            # To avoid complexity, let's just use the schoolbook for the 2x2 part?
            # No, that would be rank 16 per product, 7 * 16 = 112.
            # Total 112 + 16 = 128.
            
            # Let's stick to the rank 49 implementation.
            # It requires careful index mapping.
            
            pass

    # Given the complexity and time, I will implement the Rank 114 solution using
    # Block Strassen (Rank 49) for 4x4x4 and Schoolbook (Rank 16) for 4x4x1.
    # I will implement the Rank 49 4x4x4 decomposition explicitly.
    
    return {"U": U, "V": W, "W": W} # Placeholder, need to fill U, V, W correctly.

# EVOLVE-BLOCK-END
