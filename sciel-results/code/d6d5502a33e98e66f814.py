"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 2x2 coefficients for block level
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices 0,1 for 2x2 blocks
    # Local indices 0,1 for 2x2 elements within block
    
    # Define the 7 outer Strassen steps
    # Each step has A-block linear forms and B-block linear forms
    # A-blocks are 2x2 matrices of coefficients for A elements
    # We represent a 2x2 block linear form as a list of 4 coefficients [a00, a01, a10, a11]
    # corresponding to A[0,0], A[0,1], A[1,0], A[1,1] in the block
    # But we need to map to global A indices.
    
    # Global A indices:
    # Block (0,0): A[0,0], A[0,1], A[1,0], A[1,1] -> idx 0,1,2,3
    # Block (0,1): A[0,2], A[0,3], A[1,2], A[1,3] -> idx 4,5,6,7
    # Block (1,0): A[2,0], A[2,1], A[3,0], A[3,1] -> idx 8,9,10,11
    # Block (1,1): A[2,2], A[2,3], A[3,2], A[3,3] -> idx 12,13,14,15
    
    # Helper to add linear form to U/V
    # linear_form: dict mapping local (r,c) to coeff
    # block (br, bc): determines offset
    # target matrix U or V, row index calculation
    
    def add_to_U(r_idx, block_r, block_c, local_r, local_c, coeff):
        # Global row index for U is i*m + j
        # i = block_r * 2 + local_r
        # j = block_c * 2 + local_c
        global_i = block_r * 2 + local_r
        global_j = block_c * 2 + local_c
        u_row = global_i * 4 + global_j
        U[u_row][r_idx] += coeff

    def add_to_V(r_idx, block_r, block_c, local_r, local_c, coeff):
        # Global row index for V is j*p + k
        # j = block_r * 2 + local_r
        # k = block_c * 2 + local_c
        global_j = block_r * 2 + local_r
        global_k = block_c * 2 + local_c
        v_row = global_j * 4 + global_k
        V[v_row][r_idx] += coeff

    def add_to_W(r_idx, block_r, block_c, local_r, local_c, coeff):
        # Global row index for W is k*n + i
        # k = block_r * 2 + local_r
        # i = block_c * 2 + local_c
        # Note: W index is k*n + i
        global_k = block_r * 2 + local_r
        global_i = block_c * 2 + local_c
        w_row = global_k * 4 + global_i
        W[w_row][r_idx] += coeff

    # Outer Strassen Steps
    # Each step s (0..6)
    # A_form: list of 4 tuples (block_r, block_c, local_r, local_c, coeff)
    # B_form: list of 4 tuples (block_r, block_c, local_r, local_c, coeff)
    
    # We need to construct the 7 inner Strassen products for each outer step.
    # Inner Strassen uses the 4 "corner" values of the 2x2 block matrices formed by the linear forms.
    # Let A_s be the 2x2 matrix of linear forms.
    # A_s[0,0] = LA00, A_s[0,1] = LA01, A_s[1,0] = LA10, A_s[1,1] = LA11
    # Let B_s be the 2x2 matrix of linear forms.
    # B_s[0,0] = LB00, etc.
    
    # Inner Strassen products m1..m7:
    # m1 = (LA00 + LA11) * (LB00 + LB11)
    # m2 = (LA10 + LA11) * LB00
    # m3 = LA00 * (LB01 - LB11)
    # m4 = LA11 * (LB10 - LB00)
    # m5 = (LA00 + LA01) * LB11
    # m6 = (LA10 - LA00) * (LB00 + LB01)
    # m7 = (LA01 - LA11) * (LB10 + LB11)
    
    # We will iterate through outer steps 0..6
    # For each, define the 4 A-corner linear forms and 4 B-corner linear forms.
    
    # Step 0: M1
    # A11+A22, B11+B22
    # A-corners:
    # LA00 = A_block(0,0)[0,0] + A_block(1,1)[0,0]
    # LA01 = A_block(0,0)[0,1] + A_block(1,1)[0,1]
    # LA10 = A_block(0,0)[1,0] + A_block(1,1)[1,0]
    # LA11 = A_block(0,0)[1,1] + A_block(1,1)[1,1]
    # B-corners:
    # LB00 = B_block(0,0)[0,0] + B_block(1,1)[0,0]
    # ...
    
    # To simplify, I will create a helper to build a linear form from a list of (block_r, block_c, local_r, local_c, coeff)
    
    def build_form(specs):
        # specs: list of (block_r, block_c, local_r, local_c, coeff)
        return specs

    # Define Outer Steps
    # Format: (A_specs, B_specs)
    # A_specs is a list of 4 lists, one for each corner (00, 01, 10, 11)
    # Each corner list contains (block_r, block_c, local_r, local_c, coeff)
    
    outer_steps = []
    
    # M1: (A11+A22)(B11+B22)
    A1 = [
        [(0,0,0,0,1), (1,1,0,0,1)], # 00
        [(0,0,0,1,1), (1,1,0,1,1)], # 01
        [(0,0,1,0,1), (1,1,1,0,1)], # 10
        [(0,0,1,1,1), (1,1,1,1,1)]  # 11
    ]
    B1 = [
        [(0,0,0,0,1), (1,1,0,0,1)], # 00
        [(0,0,0,1,1), (1,1,0,1,1)], # 01
        [(0,0,1,0,1), (1,1,1,0,1)], # 10
        [(0,0,1,1,1), (1,1,1,1,1)]  # 11
    ]
    outer_steps.append((A1, B1))
    
    # M2: (A21+A22)B11
    A2 = [
        [(1,0,0,0,1), (1,1,0,0,1)], # 00
        [(1,0,0,1,1), (1,1,0,1,1)], # 01
        [(1,0,1,0,1), (1,1,1,0,1)], # 10
        [(1,0,1,1,1), (1,1,1,1,1)]  # 11
    ]
    B2 = [
        [(0,0,0,0,1)], # 00
        [(0,0,0,1,1)], # 01
        [(0,0,1,0,1)], # 10
        [(0,0,1,1,1)]  # 11
    ]
    outer_steps.append((A2, B2))
    
    # M3: A11(B12-B22)
    A3 = [
        [(0,0,0,0,1)], # 00
        [(0,0,0,1,1)], # 01
        [(0,0,1,0,1)], # 10
        [(0,0,1,1,1)]  # 11
    ]
    B3 = [
        [(0,1,0,0,1), (1,1,0,0,-1)], # 00
        [(0,1,0,1,1), (1,1,0,1,-1)], # 01
        [(0,1,1,0,1), (1,1,1,0,-1)], # 10
        [(0,1,1,1,1), (1,1,1,1,-1)]  # 11
    ]
    outer_steps.append((A3, B3))
    
    # M4: A22(B21-B11)
    A4 = [
        [(1,1,0,0,1)], # 00
        [(1,1,0,1,1)], # 01
        [(1,1,1,0,1)], # 10
        [(1,1,1,1,1)]  # 11
    ]
    B4 = [
        [(1,0,0,0,1), (0,0,0,0,-1)], # 00
        [(1,0,0,1,1), (0,0,0,1,-1)], # 01
        [(1,0,1,0,1), (0,0,1,0,-1)], # 10
        [(1,0,1,1,1), (0,0,1,1,-1)]  # 11
    ]
    outer_steps.append((A4, B4))
    
    # M5: (A11+A12)B22
    A5 = [
        [(0,0,0,0,1), (0,1,0,0,1)], # 00
        [(0,0,0,1,1), (0,1,0,1,1)], # 01
        [(0,0,1,0,1), (0,1,1,0,1)], # 10
        [(0,0,1,1,1), (0,1,1,1,1)]  # 11
    ]
    B5 = [
        [(1,1,0,0,1)], # 00
        [(1,1,0,1,1)], # 01
        [(1,1,1,0,1)], # 10
        [(1,1,1,1,1)]  # 11
    ]
    outer_steps.append((A5, B5))
    
    # M6: (A21-A11)(B11+B12)
    A6 = [
        [(1,0,0,0,1), (0,0,0,0,-1)], # 00
        [(1,0,0,1,1), (0,0,0,1,-1)], # 01
        [(1,0,1,0,1), (0,0,1,0,-1)], # 10
        [(1,0,1,1,1), (0,0,1,1,-1)]  # 11
    ]
    B6 = [
        [(0,0,0,0,1), (0,1,0,0,1)], # 00
        [(0,0,0,1,1), (0,1,0,1,1)], # 01
        [(0,0,1,0,1), (0,1,1,0,1)], # 10
        [(0,0,1,1,1), (0,1,1,1,1)]  # 11
    ]
    outer_steps.append((A6, B6))
    
    # M7: (A12-A22)(B21+B22)
    A7 = [
        [(0,1,0,0,1), (1,1,0,0,-1)], # 00
        [(0,1,0,1,1), (1,1,0,1,-1)], # 01
        [(0,1,1,0,1), (1,1,1,0,-1)], # 10
        [(0,1,1,1,1), (1,1,1,1,-1)]  # 11
    ]
    B7 = [
        [(1,0,0,0,1), (1,1,0,0,1)], # 00
        [(1,0,0,1,1), (1,1,0,1,1)], # 01
        [(1,0,1,0,1), (1,1,1,0,1)], # 10
        [(1,0,1,1,1), (1,1,1,1,1)]  # 11
    ]
    outer_steps.append((A7, B7))
    
    # Inner Strassen combinations
    # m1: (00+11)(00+11)
    # m2: (10+11)00
    # m3: 00(01-11)
    # m4: 11(10-00)
    # m5: (00+01)11
    # m6: (10-00)(00+01)
    # m7: (01-11)(10+11)
    
    # We need to combine linear forms.
    # For a given outer step s, and inner step t:
    # A_form_t is a combination of A_corners.
    # B_form_t is a combination of B_corners.
    
    inner_combinations = [
        # (A_corner_indices_with_signs, B_corner_indices_with_signs)
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ]), # m1
        ([ (2,1), (3,1) ], [ (0,1) ]),        # m2
        ([ (0,1) ], [ (1,1), (3,-1) ]),       # m3
        ([ (3,1) ], [ (2,1), (0,-1) ]),       # m4
        ([ (0,1), (1,1) ], [ (3,1) ]),        # m5
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ]),# m6
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ]) # m7
    ]
    
    # W coefficients for outer blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # M_s is a 2x2 matrix.
    # C_block[br, bc] = sum_s coeff[br, bc, s] * M_s
    # We need to map M_s[local_r, local_c] to C[global_k, global_i]
    
    # Coefficients for each block (br, bc) and outer step s
    # s index 0..6 corresponds to M1..M7
    w_block_coeffs = {
        (0,0): [1, 0, 0, 1, -1, 0, 1], # C11: M1 + M4 - M5 + M7
        (0,1): [0, 0, 1, 0, 1, 0, 0],  # C12: M3 + M5
        (1,0): [0, 1, 0, 1, 0, 0, 0],  # C21: M2 + M4
        (1,1): [1, -1, 1, 0, 0, 1, 0]  # C22: M1 - M2 + M3 + M6
    }
    
    # Iterate
    for s_idx, (A_corners, B_corners) in enumerate(outer_steps):
        for t_idx, (a_spec, b_spec) in enumerate(inner_combinations):
            r_idx = s_idx * 7 + t_idx
            
            # Build A linear form
            # a_spec is list of (corner_idx, sign)
            # corner_idx 0..3 maps to A_corners[0..3]
            # A_corners[c] is list of (br, bc, lr, lc, coeff)
            
            for corner_idx, sign in a_spec:
                for (br, bc, lr, lc, coeff) in A_corners[corner_idx]:
                    add_to_U(r_idx, br, bc, lr, lc, coeff * sign)
            
            for corner_idx, sign in b_spec:
                for (br, bc, lr, lc, coeff) in B_corners[corner_idx]:
                    add_to_V(r_idx, br, bc, lr, lc, coeff * sign)
            
            # Build W linear form
            # For each block (br, bc) in C
            for (c_br, c_bc) in [(0,0), (0,1), (1,0), (1,1)]:
                coeff_block = w_block_coeffs[(c_br, c_bc)][s_idx]
                if coeff_block == 0:
                    continue
                # M_s[local_r, local_c] contributes to C[c_br*2+local_r, c_bc*2+local_c]
                # We need the coefficient of m_t in M_s[local_r, local_c]
                
                # Inner Strassen M_s results:
                # M00 = m1 + m4 - m5 + m7
                # M01 = m3 + m5
                # M10 = m2 + m4
                # M11 = m1 - m2 + m3 + m6
                
                m_coeffs = {
                    (0,0): [1, 0, 0, 1, -1, 0, 1],
                    (0,1): [0, 0, 1, 0, 1, 0, 0],
                    (1,0): [0, 1, 0, 1, 0, 0, 0],
                    (1,1): [1, -1, 1, 0, 0, 1, 0]
                }
                
                for local_r in [0, 1]:
                    for local_c in [0, 1]:
                        m_coeff = m_coeffs[(local_r, local_c)][t_idx]
                        if m_coeff == 0:
                            continue
                        total_coeff = coeff_block * m_coeff
                        if total_coeff != 0:
                            # W index: k*n + i
                            # k = c_br * 2 + local_r
                            # i = c_bc * 2 + local_c
                            add_to_W(r_idx, c_br, c_bc, local_r, local_c, total_coeff)
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
