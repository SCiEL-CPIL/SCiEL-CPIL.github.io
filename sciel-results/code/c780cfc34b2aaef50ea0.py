import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 terms definitions
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Contributions of M1..M7 to Z = XY
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Block products P1..P7
    # P1: X=A00+A11, Y=B00+B11
    # P2: X=A10+A11, Y=B00
    # P3: X=A00, Y=B01-B11
    # P4: X=A11, Y=B10-B00
    # P5: X=A00+A01, Y=B11
    # P6: X=A10-A00, Y=B00+B01
    # P7: X=A01-A11, Y=B10+B11
    
    # Contributions of P1..P7 to C blocks
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Helper to get A indices for a block expression
    # A blocks: 00: (0,0)-(1,1), 01: (0,2)-(1,3), 10: (2,0)-(3,1), 11: (2,2)-(3,3)
    def get_A_indices(block_ids, signs):
        # block_ids: list of block indices (0,1,2,3)
        # signs: list of +1/-1
        indices = {}
        for b_id, sign in zip(block_ids, signs):
            if b_id == 0: # A00
                for i in range(2):
                    for j in range(2):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 1: # A01
                for i in range(2):
                    for j in range(2, 4):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 2: # A10
                for i in range(2, 4):
                    for j in range(2):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 3: # A11
                for i in range(2, 4):
                    for j in range(2, 4):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
        return indices

    def get_B_indices(block_ids, signs):
        indices = {}
        for b_id, sign in zip(block_ids, signs):
            if b_id == 0: # B00
                for i in range(2):
                    for j in range(2):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 1: # B01
                for i in range(2):
                    for j in range(2, 4):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 2: # B10
                for i in range(2, 4):
                    for j in range(2):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
            elif b_id == 3: # B11
                for i in range(2, 4):
                    for j in range(2, 4):
                        idx = i*4+j
                        indices[idx] = indices.get(idx, 0) + sign
        return indices

    # Define the 7 block products' X and Y in terms of blocks
    # Each X/Y is a 2x2 matrix of block expressions
    # We represent a 2x2 matrix of block expressions as a list of 4 lists of (block_ids, signs)
    
    block_products = [
        # P1
        {
            'X': [[ [0, 3], [1, 1] ], [ [0, 3], [1, 1] ], [ [0, 3], [1, 1] ], [ [0, 3], [1, 1] ] ], # A00+A11 for all entries? No.
            # Wait, X is a 2x2 matrix.
            # P1: X = A00 + A11. This means the entire 2x2 matrix X is A00+A11.
            # So X00 = A00+A11, X01 = A00+A11, etc? No.
            # A00 is a 2x2 block. A11 is a 2x2 block.
            # X = A00 + A11 means X_{ij} = (A00)_{ij} + (A11)_{ij}.
            # So for X00 (top-left of X), it corresponds to (A00)00 + (A11)00.
            # My get_A_indices takes block IDs.
            # For X00, we need the top-left entry of the block sum.
            # This logic is getting slightly tangled. Let's simplify.
            
            # Let's define X and Y entries directly as linear combos of A/B scalar indices.
        },
    ]
    
    # Re-approach: Define X and Y entries for each P_k explicitly as linear combos.
    
    # P1: X = A00+A11, Y = B00+B11
    # X00 = A00,00 + A11,00 -> A[0,0] + A[2,2]
    # X01 = A00,01 + A11,01 -> A[0,1] + A[2,3]
    # X10 = A00,10 + A11,10 -> A[1,0] + A[3,2]
    # X11 = A00,11 + A11,11 -> A[1,1] + A[3,3]
    
    # It is safer to generate the indices programmatically for each block product.
    
    def get_block_entry_indices(block_id, r, c):
        # Returns list of (idx, sign) for the entry (r,c) of the block block_id
        # r,c in 0..1
        if block_id == 0: # 00
            return [((r)*4 + c, 1)]
        elif block_id == 1: # 01
            return [((r)*4 + (c+2), 1)]
        elif block_id == 2: # 10
            return [((r+2)*4 + c, 1)]
        elif block_id == 3: # 11
            return [((r+2)*4 + (c+2), 1)]

    # Define X and Y for each P_k as 2x2 matrices of linear forms
    # Linear form is a dict {index: coeff}
    
    def add_forms(f1, f2):
        res = dict(f1)
        for k, v in f2.items():
            res[k] = res.get(k, 0) + v
        return res

    def scale_form(f, s):
        return {k: v*s for k, v in f.items()}

    def get_block_form(block_id, r, c):
        idxs = get_block_entry_indices(block_id, r, c)
        return {i: s for i, s in idxs}

    # Precompute block forms for A and B
    A_forms = {}
    B_forms = {}
    for b in range(4):
        for r in range(2):
            for c in range(2):
                A_forms[(b,r,c)] = get_block_form(b, r, c)
                B_forms[(b,r,c)] = get_block_form(b, r, c)

    # Define the 7 P_k
    # Each P_k has X (2x2 of forms) and Y (2x2 of forms)
    P_defs = []
    
    # P1
    X = [[add_forms(A_forms[(0,0,0)], A_forms[(3,0,0)]), add_forms(A_forms[(0,0,1)], A_forms[(3,0,1)]),
          add_forms(A_forms[(0,1,0)], A_forms[(3,1,0)]), add_forms(A_forms[(0,1,1)], A_forms[(3,1,1)])]]
    # Wait, X is 2x2.
    X = [
        [add_forms(A_forms[(0,0,0)], A_forms[(3,0,0)]), add_forms(A_forms[(0,0,1)], A_forms[(3,0,1)])],
        [add_forms(A_forms[(0,1,0)], A_forms[(3,1,0)]), add_forms(A_forms[(0,1,1)], A_forms[(3,1,1)])]
    ]
    Y = [
        [add_forms(B_forms[(0,0,0)], B_forms[(3,0,0)]), add_forms(B_forms[(0,0,1)], B_forms[(3,0,1)])],
        [add_forms(B_forms[(0,1,0)], B_forms[(3,1,0)]), add_forms(B_forms[(0,1,1)], B_forms[(3,1,1)])]
    ]
    P_defs.append((X, Y))
    
    # P2: X=A10+A11, Y=B00
    X = [
        [add_forms(A_forms[(2,0,0)], A_forms[(3,0,0)]), add_forms(A_forms[(2,0,1)], A_forms[(3,0,1)])],
        [add_forms(A_forms[(2,1,0)], A_forms[(3,1,0)]), add_forms(A_forms[(2,1,1)], A_forms[(3,1,1)])]
    ]
    Y = [
        [B_forms[(0,0,0)], B_forms[(0,0,1)]],
        [B_forms[(0,1,0)], B_forms[(0,1,1)]]
    ]
    P_defs.append((X, Y))
    
    # P3: X=A00, Y=B01-B11
    X = [
        [A_forms[(0,0,0)], A_forms[(0,0,1)]],
        [A_forms[(0,1,0)], A_forms[(0,1,1)]]
    ]
    Y = [
        [add_forms(B_forms[(1,0,0)], scale_form(B_forms[(3,0,0)], -1)), add_forms(B_forms[(1,0,1)], scale_form(B_forms[(3,0,1)], -1))],
        [add_forms(B_forms[(1,1,0)], scale_form(B_forms[(3,1,0)], -1)), add_forms(B_forms[(1,1,1)], scale_form(B_forms[(3,1,1)], -1))]
    ]
    P_defs.append((X, Y))
    
    # P4: X=A11, Y=B10-B00
    X = [
        [A_forms[(3,0,0)], A_forms[(3,0,1)]],
        [A_forms[(3,1,0)], A_forms[(3,1,1)]]
    ]
    Y = [
        [add_forms(B_forms[(2,0,0)], scale_form(B_forms[(0,0,0)], -1)), add_forms(B_forms[(2,0,1)], scale_form(B_forms[(0,0,1)], -1))],
        [add_forms(B_forms[(2,1,0)], scale_form(B_forms[(0,1,0)], -1)), add_forms(B_forms[(2,1,1)], scale_form(B_forms[(0,1,1)], -1))]
    ]
    P_defs.append((X, Y))
    
    # P5: X=A00+A01, Y=B11
    X = [
        [add_forms(A_forms[(0,0,0)], A_forms[(1,0,0)]), add_forms(A_forms[(0,0,1)], A_forms[(1,0,1)])],
        [add_forms(A_forms[(0,1,0)], A_forms[(1,1,0)]), add_forms(A_forms[(0,1,1)], A_forms[(1,1,1)])]
    ]
    Y = [
        [B_forms[(3,0,0)], B_forms[(3,0,1)]],
        [B_forms[(3,1,0)], B_forms[(3,1,1)]]
    ]
    P_defs.append((X, Y))
    
    # P6: X=A10-A00, Y=B00+B01
    X = [
        [add_forms(A_forms[(2,0,0)], scale_form(A_forms[(0,0,0)], -1)), add_forms(A_forms[(2,0,1)], scale_form(A_forms[(0,0,1)], -1))],
        [add_forms(A_forms[(2,1,0)], scale_form(A_forms[(0,1,0)], -1)), add_forms(A_forms[(2,1,1)], scale_form(A_forms[(0,1,1)], -1))]
    ]
    Y = [
        [add_forms(B_forms[(0,0,0)], B_forms[(1,0,0)]), add_forms(B_forms[(0,0,1)], B_forms[(1,0,1)])],
        [add_forms(B_forms[(0,1,0)], B_forms[(1,1,0)]), add_forms(B_forms[(0,1,1)], B_forms[(1,1,1)])]
    ]
    P_defs.append((X, Y))
    
    # P7: X=A01-A11, Y=B10+B11
    X = [
        [add_forms(A_forms[(1,0,0)], scale_form(A_forms[(3,0,0)], -1)), add_forms(A_forms[(1,0,1)], scale_form(A_forms[(3,0,1)], -1))],
        [add_forms(A_forms[(1,1,0)], scale_form(A_forms[(3,1,0)], -1)), add_forms(A_forms[(1,1,1)], scale_form(A_forms[(3,1,1)], -1))]
    ]
    Y = [
        [add_forms(B_forms[(2,0,0)], B_forms[(3,0,0)]), add_forms(B_forms[(2,0,1)], B_forms[(3,0,1)])],
        [add_forms(B_forms[(2,1,0)], B_forms[(3,1,0)]), add_forms(B_forms[(2,1,1)], B_forms[(3,1,1)])]
    ]
    P_defs.append((X, Y))

    # C Block Contributions
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # P_index: 0..6
    C_block_contribs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00 gets P1, P4, -P5, P7
        {2: 1, 4: 1},               # C01 gets P3, P5
        {1: 1, 3: 1},               # C10 gets P2, P4
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11 gets P1, -P2, P3, P6
    ]

    # Strassen Scalar Terms
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z contributions
    # Z00: M1, M4, -M5, M7
    # Z01: M3, M5
    # Z10: M2, M4
    # Z11: M1, -M2, M3, M6
    
    Z_contribs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # Z00
        {2: 1, 4: 1},               # Z01
        {1: 1, 3: 1},               # Z10
        {0: 1, 1: -1, 2: 1, 5: 1}  # Z11
    ]

    col = 0
    for k in range(7):
        X, Y = P_defs[k]
        x00, x01 = X[0][0], X[0][1]
        x10, x11 = X[1][0], X[1][1]
        y00, y01 = Y[0][0], Y[0][1]
        y10, y11 = Y[1][0], Y[1][1]
        
        terms = [
            (add_forms(x00, x11), add_forms(y00, y11)), # M1
            (add_forms(x10, x11), y00),                 # M2
            (x00, add_forms(y01, scale_form(y11, -1))), # M3
            (x11, add_forms(y10, scale_form(y00, -1))), # M4
            (add_forms(x00, x01), y11),                 # M5
            (add_forms(x10, scale_form(x00, -1)), add_forms(y00, y01)), # M6
            (add_forms(x01, scale_form(x11, -1)), add_forms(y10, y11))  # M7
        ]
        
        for s in range(7):
            U_form, V_form = terms[s]
            
            # For each Z entry this term affects
            for zr in range(2):
                for zc in range(2):
                    if s not in Z_contribs[zr*2+zc]:
                        continue
                    delta = Z_contribs[zr*2+zc][s]
                    
                    # For each C block this P_k affects
                    for cb_r in range(2):
                        for cb_c in range(2):
                            cb_idx = cb_r*2+cb_c
                            if k not in C_block_contribs[cb_idx]:
                                continue
                            epsilon = C_block_contribs[cb_idx][k]
                            
                            gamma = delta * epsilon
                            
                            # Final C entry indices
                            row = cb_r*2 + zr
                            col_c = cb_c*2 + zc
                            
                            # Update U, V, W
                            for idx_a, coeff_a in U_form.items():
                                U[idx_a][col] += coeff_a
                            for idx_b, coeff_b in V_form.items():
                                V[idx_b][col] += coeff_b
                            
                            # W index: k*n+i -> col_c*4 + row
                            W[col_c*4 + row][col] += gamma
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
