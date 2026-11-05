"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block partition 4 + 1
    # Rank 110 decomposition
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        # u_dict: {index: coeff}
        # v_dict: {index: coeff}
        # w_dict: {index: coeff}
        terms.append((u_dict, v_dict, w_dict))

    def idx_A(i, j):
        return i * m + j

    def idx_B(i, j):
        return i * p + j

    def idx_C(i, j):
        return i * n + j

    # 1. 4x4x4 block (Top-Left) using Strassen (Rank 49)
    # A: rows 0-3, cols 0-3
    # B: rows 0-3, cols 0-3
    # C: rows 0-3, cols 0-3
    
    # Define 2x2 block index ranges
    # TL: r 0-1, c 0-1
    # TR: r 0-1, c 2-3
    # BL: r 2-3, c 0-1
    # BR: r 2-3, c 2-3
    
    def get_2x2_indices(base_r, base_c, matrix_type):
        # Returns 4 indices: (00, 01, 10, 11)
        if matrix_type == 'A':
            return [idx_A(base_r, base_c), idx_A(base_r, base_c+1), idx_A(base_r+1, base_c), idx_A(base_r+1, base_c+1)]
        elif matrix_type == 'B':
            return [idx_B(base_r, base_c), idx_B(base_r, base_c+1), idx_B(base_r+1, base_c), idx_B(base_r+1, base_c+1)]
        elif matrix_type == 'C':
            return [idx_C(base_r, base_c), idx_C(base_r, base_c+1), idx_C(base_r+1, base_c), idx_C(base_r+1, base_c+1)]

    # Strassen for 2x2
    # Inputs: A_idx, B_idx, C_idx (lists of 4 indices)
    def strassen_2x2(A_idx, B_idx, C_idx):
        # A_idx: [a00, a01, a10, a11]
        # B_idx: [b00, b01, b10, b11]
        # C_idx: [c00, c01, c10, c11]
        
        # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        add_term(
            {A_idx[0]: 1, A_idx[3]: 1},
            {B_idx[0]: 1, B_idx[3]: 1},
            {C_idx[0]: 1, C_idx[3]: 1}
        )
        
        # M2 = (a10+a11)b00 -> C10 += 1
        add_term(
            {A_idx[2]: 1, A_idx[3]: 1},
            {B_idx[0]: 1},
            {C_idx[2]: 1}
        )
        
        # M3 = a00(b01-b11) -> C01 += 1
        add_term(
            {A_idx[0]: 1},
            {B_idx[1]: 1, B_idx[3]: -1},
            {C_idx[1]: 1}
        )
        
        # M4 = a11(b10-b00) -> C00 += 1, C10 += 1
        add_term(
            {A_idx[3]: 1},
            {B_idx[2]: 1, B_idx[0]: -1},
            {C_idx[0]: 1, C_idx[2]: 1}
        )
        
        # M5 = (a00+a01)b11 -> C00 -= 1, C01 += 1
        add_term(
            {A_idx[0]: 1, A_idx[1]: 1},
            {B_idx[3]: 1},
            {C_idx[0]: -1, C_idx[1]: 1}
        )
        
        # M6 = (a10-a00)(b00+b01) -> C11 += 1
        add_term(
            {A_idx[2]: 1, A_idx[0]: -1},
            {B_idx[0]: 1, B_idx[1]: 1},
            {C_idx[3]: 1}
        )
        
        # M7 = (a01-a11)(b10+b11) -> C00 += 1, C11 -= 1
        add_term(
            {A_idx[1]: 1, A_idx[3]: -1},
            {B_idx[2]: 1, B_idx[3]: 1},
            {C_idx[0]: 1, C_idx[3]: -1}
        )

    # Now build the 4x4 Strassen
    # Get indices for 4 blocks
    A_TL = get_2x2_indices(0, 0, 'A')
    A_TR = get_2x2_indices(0, 2, 'A')
    A_BL = get_2x2_indices(2, 0, 'A')
    A_BR = get_2x2_indices(2, 2, 'A')
    
    B_TL = get_2x2_indices(0, 0, 'B')
    B_TR = get_2x2_indices(0, 2, 'B')
    B_BL = get_2x2_indices(2, 0, 'B')
    B_BR = get_2x2_indices(2, 2, 'B')
    
    C_TL = get_2x2_indices(0, 0, 'C')
    C_TR = get_2x2_indices(0, 2, 'C')
    C_BL = get_2x2_indices(2, 0, 'C')
    C_BR = get_2x2_indices(2, 2, 'C')

    # Helper to combine 2x2 index lists into linear combinations
    def combine_indices(list1, list2, coeff1=1, coeff2=1):
        # Returns a dict {idx: coeff}
        d = {}
        for idx in list1:
            d[idx] = d.get(idx, 0) + coeff1
        for idx in list2:
            d[idx] = d.get(idx, 0) + coeff2
        return d

    # M1 = (A_TL + A_BR)(B_TL + B_BR)
    A_M1 = combine_indices(A_TL, A_BR)
    B_M1 = combine_indices(B_TL, B_BR)
    # The result of M1 is a 2x2 matrix.
    # We need to apply Strassen to this 2x2 multiplication.
    # But strassen_2x2 expects lists of 4 indices.
    # Here, the "indices" are actually linear combinations.
    # We need a more general function.
    
    # Let's refactor: Instead of hardcoding strassen_2x2 with indices,
    # we will generate the 7 terms for a 2x2 multiplication given
    # the linear combinations for A and B elements.
    
    def strassen_2x2_general(A_elems, B_elems, C_elems):
        # A_elems: dict {pos: {idx: coeff}} where pos is 0,1,2,3 (00,01,10,11)
        # B_elems: dict {pos: {idx: coeff}}
        # C_elems: dict {pos: {idx: coeff}}
        
        # M1 = (A00+A11)(B00+B11)
        A_M1 = combine_dicts(A_elems[0], A_elems[3])
        B_M1 = combine_dicts(B_elems[0], B_elems[3])
        # C contribution: C00 += 1, C11 += 1
        C_M1 = {0: 1, 3: 1}
        add_term_general(A_M1, B_M1, C_M1, C_elems)

        # M2 = (A10+A11)B00
        A_M2 = combine_dicts(A_elems[2], A_elems[3])
        B_M2 = B_elems[0]
        C_M2 = {2: 1}
        add_term_general(A_M2, B_M2, C_M2, C_elems)

        # M3 = A00(B01-B11)
        A_M3 = A_elems[0]
        B_M3 = combine_dicts(B_elems[1], B_elems[3], 1, -1)
        C_M3 = {1: 1}
        add_term_general(A_M3, B_M3, C_M3, C_elems)

        # M4 = A11(B10-B00)
        A_M4 = A_elems[3]
        B_M4 = combine_dicts(B_elems[2], B_elems[0], 1, -1)
        C_M4 = {0: 1, 2: 1}
        add_term_general(A_M4, B_M4, C_M4, C_elems)

        # M5 = (A00+A01)B11
        A_M5 = combine_dicts(A_elems[0], A_elems[1])
        B_M5 = B_elems[3]
        C_M5 = {0: -1, 1: 1}
        add_term_general(A_M5, B_M5, C_M5, C_elems)

        # M6 = (A10-A00)(B00+B01)
        A_M6 = combine_dicts(A_elems[2], A_elems[0], 1, -1)
        B_M6 = combine_dicts(B_elems[0], B_elems[1])
        C_M6 = {3: 1}
        add_term_general(A_M6, B_M6, C_M6, C_elems)

        # M7 = (A01-A11)(B10+B11)
        A_M7 = combine_dicts(A_elems[1], A_elems[3], 1, -1)
        B_M7 = combine_dicts(B_elems[2], B_elems[3])
        C_M7 = {0: 1, 3: -1}
        add_term_general(A_M7, B_M7, C_M7, C_elems)

    def combine_dicts(d1, d2, c1=1, c2=1):
        d = {}
        for k, v in d1.items():
            d[k] = d.get(k, 0) + v * c1
        for k, v in d2.items():
            d[k] = d.get(k, 0) + v * c2
        return d

    def add_term_general(u_dict, v_dict, c_contrib, C_elems):
        # c_contrib: {pos: coeff}
        # C_elems: {pos: {idx: coeff}}
        w_dict = {}
        for pos, coeff in c_contrib.items():
            if coeff != 0:
                c_dict = C_elems[pos]
                for idx, c_val in c_dict.items():
                    w_dict[idx] = w_dict.get(idx, 0) + coeff * c_val
        add_term(u_dict, v_dict, w_dict)

    # Prepare A_elems, B_elems, C_elems for the 4 2x2 blocks
    def make_elems(indices):
        return {
            0: {indices[0]: 1},
            1: {indices[1]: 1},
            2: {indices[2]: 1},
            3: {indices[3]: 1}
        }

    A_TL_e = make_elems(A_TL)
    A_TR_e = make_elems(A_TR)
    A_BL_e = make_elems(A_BL)
    A_BR_e = make_elems(A_BR)
    
    B_TL_e = make_elems(B_TL)
    B_TR_e = make_elems(B_TR)
    B_BL_e = make_elems(B_BL)
    B_BR_e = make_elems(B_BR)
    
    C_TL_e = make_elems(C_TL)
    C_TR_e = make_elems(C_TR)
    C_BL_e = make_elems(C_BL)
    C_BR_e = make_elems(C_BR)

    # M1 = (A_TL + A_BR)(B_TL + B_BR) -> Result is 2x2 matrix at C_TL?
    # Wait, the result of M1 is a 2x2 matrix.
    # In the 4x4 Strassen assembly:
    # C_TL = M1 + M4 - M5 + M7
    # C_TR = M3 + M5
    # C_BL = M2 + M4
    # C_BR = M1 - M2 + M3 + M6
    
    # So, for M1, the "C_elems" passed to strassen_2x2_general should be C_TL_e?
    # No. The output of the 2x2 multiplication M1 is a 2x2 matrix.
    # This matrix is then added to C_TL and C_BR.
    # So the linear combination for the output of M1 is:
    # C_out = 1 * C_TL + 1 * C_BR?
    # No, the multiplication M1 produces a matrix $P_1$.
    # $C_{TL} += P_1$ and $C_{BR} += P_1$.
    # So the $C$ indices for the rank-1 terms of M1 should map to both $C_{TL}$ and $C_{BR}$ blocks.
    
    # Let's define the output mapping for each meta-term.
    # M1 contributes to C_TL (+1) and C_BR (+1).
    # M2 contributes to C_BL (+1).
    # M3 contributes to C_TR (+1).
    # M4 contributes to C_TL (+1) and C_BL (+1).
    # M5 contributes to C_TL (-1) and C_TR (+1).
    # M6 contributes to C_BR (+1).
    # M7 contributes to C_TL (+1) and C_BR (-1).

    def get_C_elems_for_meta(M_id):
        # Returns dict {pos: {idx: coeff}} for the 2x2 output block
        # The output block is added to the global C.
        # We need to map the 2x2 output positions (0,1,2,3) to global C indices.
        # If M1 contributes to C_TL and C_BR, then:
        # Output pos 0 (top-left of 2x2) adds to C_TL[0] and C_BR[0].
        # Output pos 1 (top-right) adds to C_TL[1] and C_BR[1].
        # etc.
        
        # This logic is getting complicated.
        # Simpler: Just expand the 7 meta-terms into 7 scalar terms each,
        # where the scalar terms directly target the global C indices.
        
        pass

    # I will implement the 7 meta-terms directly.
    # M1 = (A_TL + A_BR)(B_TL + B_BR)
    # A_M1 = A_TL + A_BR
    # B_M1 = B_TL + B_BR
    # The product is a 2x2 matrix P.
    # P is computed via Strassen (7 terms).
    # Each term of P is added to C_TL and C_BR.
    
    # I will call strassen_2x2_general with A_M1, B_M1, and a special C_elems
    # that maps to both C_TL and C_BR.
    
    def make_dual_C_elems(C1, C2, c1_coeff, c2_coeff):
        # C1, C2 are dicts {pos: {idx: coeff}}
        # Returns dict {pos: {idx: coeff}}
        d = {}
        for pos in range(4):
            d[pos] = {}
            for idx, val in C1[pos].items():
                d[pos][idx] = d[pos].get(idx, 0) + val * c1_coeff
            for idx, val in C2[pos].items():
                d[pos][idx] = d[pos].get(idx, 0) + val * c2_coeff
        return d

    # M1
    A_M1 = combine_elems(A_TL_e, A_BR_e)
    B_M1 = combine_elems(B_TL_e, B_BR_e)
    C_M1 = make_dual_C_elems(C_TL_e, C_BR_e, 1, 1)
    strassen_2x2_general(A_M1, B_M1, C_M1)
    
    # M2
    A_M2 = combine_elems(A_BL_e, A_BR_e)
    B_M2 = B_TL_e
    C_M2 = make_dual_C_elems(C_BL_e, C_BL_e, 1, 0) # Only C_BL
    strassen_2x2_general(A_M2, B_M2, C_M2)
    
    # M3
    A_M3 = A_TL_e
    B_M3 = combine_elems(B_TR_e, B_BR_e, 1, -1)
    C_M3 = make_dual_C_elems(C_TR_e, C_TR_e, 1, 0) # Only C_TR
    strassen_2x2_general(A_M3, B_M3, C_M3)
    
    # M4
    A_M4 = A_BR_e
    B_M4 = combine_elems(B_BL_e, B_TL_e, 1, -1)
    C_M4 = make_dual_C_elems(C_TL_e, C_BL_e, 1, 1)
    strassen_2x2_general(A_M4, B_M4, C_M4)
    
    # M5
    A_M5 = combine_elems(A_TL_e, A_TR_e)
    B_M5 = B_BR_e
    C_M5 = make_dual_C_elems(C_TL_e, C_TR_e, -1, 1)
    strassen_2x2_general(A_M5, B_M5, C_M5)
    
    # M6
    A_M6 = combine_elems(A_BL_e, A_TL_e, 1, -1)
    B_M6 = combine_elems(B_TL_e, B_TR_e)
    C_M6 = make_dual_C_elems(C_BR_e, C_BR_e, 1, 0)
    strassen_2x2_general(A_M6, B_M6, C_M6)
    
    # M7
    A_M7 = combine_elems(A_TR_e, A_BR_e, 1, -1)
    B_M7 = combine_elems(B_BL_e, B_BR_e)
    C_M7 = make_dual_C_elems(C_TL_e, C_BR_e, 1, -1)
    strassen_2x2_general(A_M7, B_M7, C_M7)

    # 2. Schoolbook terms for the rest
    # A: 5x5, B: 5x5, C: 5x5
    
    # 4x1x4: A[0:4, 4] B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            add_term(
                {idx_A(i, 4): 1},
                {idx_B(4, j): 1},
                {idx_C(i, j): 1}
            )
            
    # 4x4x1: A[0:4, 0:4] B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            add_term(
                {idx_A(i, j): 1},
                {idx_B(j, 4): 1},
                {idx_C(i, 4): 1}
            )
            
    # 4x1x1: A[0:4, 4] B[4, 4] -> C[0:4, 4]
    for i in range(4):
        add_term(
            {idx_A(i, 4): 1},
            {idx_B(4, 4): 1},
            {idx_C(i, 4): 1}
        )
        
    # 1x4x4: A[4, 0:4] B[0:4, 0:4] -> C[4, 0:4]
    for j in range(4):
        for k in range(4):
            add_term(
                {idx_A(4, j): 1},
                {idx_B(j, k): 1},
                {idx_C(4, k): 1}
            )
            
    # 1x1x4: A[4, 4] B[4, 0:4] -> C[4, 0:4]
    for j in range(4):
        add_term(
            {idx_A(4, 4): 1},
            {idx_B(4, j): 1},
            {idx_C(4, j): 1}
        )
        
    # 1x4x1: A[4, 0:4] B[0:4, 4] -> C[4, 4]
    for j in range(4):
        add_term(
            {idx_A(4, j): 1},
            {idx_B(j, 4): 1},
            {idx_C(4, 4): 1}
        )
        
    # 1x1x1: A[4, 4] B[4, 4] -> C[4, 4]
    add_term(
        {idx_A(4, 4): 1},
        {idx_B(4, 4): 1},
        {idx_C(4, 4): 1}
    )

    # Construct U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
            
    return {"U": U, "V": V, "W": W}

# Helper functions defined inside construct or globally?
# I defined them inside construct in the thought process, but they need to be accessible.
# I will move helper functions inside construct to avoid namespace pollution or just define them before.

# EVOLVE-BLOCK-END
