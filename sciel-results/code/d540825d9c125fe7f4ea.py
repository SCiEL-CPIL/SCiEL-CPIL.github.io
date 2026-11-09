import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 construction
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(U_idx, V_idx, W_idx, coeff):
        nonlocal col
        if coeff == 0:
            return
        U[U_idx][col] += coeff
        V[V_idx][col] += coeff
        W[W_idx][col] += coeff
        col += 1

    # Helper for Strassen 2x2x2
    # X is 2x2, Y is 2x2, Z is 2x2
    # Indices for X: (0,0), (0,1), (1,0), (1,1)
    # Indices for Y: (0,0), (0,1), (1,0), (1,1)
    # Indices for Z: (0,0), (0,1), (1,0), (1,1)
    def strassen_block(U_base, V_base, W_base):
        # U_base: list of 4 U indices for X
        # V_base: list of 4 V indices for Y
        # W_base: list of 4 W indices for Z
        # X00, X01, X10, X11
        x00, x01, x10, x11 = U_base
        y00, y01, y10, y11 = V_base
        z00, z01, z10, z11 = W_base
        
        # M1 = (X00+X11)(Y00+Y11)
        # We need to expand (X00+X11) and (Y00+Y11) into linear combinations
        # This requires multiple terms in the tensor decomposition
        # Term 1: X00 * Y00
        # Term 2: X00 * Y11
        # Term 3: X11 * Y00
        # Term 4: X11 * Y11
        # But Strassen combines them.
        # The tensor decomposition allows any linear combination.
        # So we can just add the product of the linear forms.
        
        # M1
        # (X00 + X11) * (Y00 + Y11)
        # This is one term in the rank decomposition?
        # No, the rank decomposition is a sum of products of linear forms.
        # So M1 is ONE term.
        # U_part: X00: 1, X11: 1
        # V_part: Y00: 1, Y11: 1
        # W_part: Z00: 1, Z01: 0, Z10: 0, Z11: 1 (from M1 contribution to Z00, Z11? No)
        
        # Let's trace the contributions to Z.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We will add 7 terms.
        
        # Term 1: M1
        # U: x00=1, x11=1
        # V: y00=1, y11=1
        # W: z00=1, z11=1
        add_term_block(x00, x11, y00, y11, z00, z11, 1, 1, 1, 1)
        
        # Term 2: M2
        # M2 = (X10+X11)Y00
        # U: x10=1, x11=1
        # V: y00=1
        # W: z10=1, z11=-1
        add_term_block(x10, x11, y00, None, z10, z11, 1, 1, 1, -1)
        
        # Term 3: M3
        # M3 = X00(Y10-Y11)
        # U: x00=1
        # V: y10=1, y11=-1
        # W: z01=1, z11=1
        add_term_block(x00, None, y10, y11, z01, z11, 1, 1, 1, 1)
        
        # Term 4: M4
        # M4 = X11(Y01-Y00)
        # U: x11=1
        # V: y01=1, y00=-1
        # W: z00=1, z10=1
        add_term_block(x11, None, y01, y00, z00, z10, 1, 1, 1, 1)
        
        # Term 5: M5
        # M5 = (X00+X01)Y11
        # U: x00=1, x01=1
        # V: y11=1
        # W: z01=1, z00=-1
        add_term_block(x00, x01, y11, None, z01, z00, 1, 1, 1, -1)
        
        # Term 6: M6
        # M6 = (X10-X00)(Y00+Y10)
        # U: x10=1, x00=-1
        # V: y00=1, y10=1
        # W: z11=1
        add_term_block(x10, x00, y00, y10, z11, None, 1, -1, 1, 1)
        
        # Term 7: M7
        # M7 = (X01-X11)(Y11+Y01)
        # U: x01=1, x11=-1
        # V: y11=1, y01=1
        # W: z00=1
        add_term_block(x01, x11, y11, y01, z00, None, 1, -1, 1, 1)

    def add_term_block(u1, u2, v1, v2, w1, w2, cu1, cu2, cv1, cv2):
        # This helper is tricky because we need to add to multiple U, V, W indices for a single term
        # But my add_term only takes one index.
        # I need a more flexible add_term.
        pass

    # Let's rewrite the addition logic to handle linear combinations properly.
    # A term in the decomposition is defined by:
    # U_vec (length n*m), V_vec (length m*p), W_vec (length p*n)
    # such that T = sum (U_vec . A) (V_vec . B) (W_vec . C)
    
    # I will create a list of (U_dict, V_dict, W_dict)
    terms = []
    
    def add_strassen(u_indices, v_indices, w_indices):
        # u_indices: [x00, x01, x10, x11]
        # v_indices: [y00, y01, y10, y11]
        # w_indices: [z00, z01, z10, z11]
        
        x00, x01, x10, x11 = u_indices
        y00, y01, y10, y11 = v_indices
        z00, z01, z10, z11 = w_indices
        
        # M1
        terms.append({
            'U': {x00: 1, x11: 1},
            'V': {y00: 1, y11: 1},
            'W': {z00: 1, z11: 1}
        })
        # M2
        terms.append({
            'U': {x10: 1, x11: 1},
            'V': {y00: 1},
            'W': {z10: 1, z11: -1}
        })
        # M3
        terms.append({
            'U': {x00: 1},
            'V': {y10: 1, y11: -1},
            'W': {z01: 1, z11: 1}
        })
        # M4
        terms.append({
            'U': {x11: 1},
            'V': {y01: 1, y00: -1},
            'W': {z00: 1, z10: 1}
        })
        # M5
        terms.append({
            'U': {x00: 1, x01: 1},
            'V': {y11: 1},
            'W': {z01: 1, z00: -1}
        })
        # M6
        terms.append({
            'U': {x10: 1, x00: -1},
            'V': {y00: 1, y10: 1},
            'W': {z11: 1}
        })
        # M7
        terms.append({
            'U': {x01: 1, x11: -1},
            'V': {y11: 1, y01: 1},
            'W': {z00: 1}
        })

    def add_schoolbook_1x2x2(u_indices, v_indices, w_indices):
        # u: [a0, a1]
        # v: [b00, b01, b10, b11]
        # w: [c0, c1]
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        a0, a1 = u_indices
        b00, b01, b10, b11 = v_indices
        c0, c1 = w_indices
        
        terms.append({'U': {a0: 1}, 'V': {b00: 1}, 'W': {c0: 1}})
        terms.append({'U': {a1: 1}, 'V': {b10: 1}, 'W': {c0: 1}})
        terms.append({'U': {a0: 1}, 'V': {b01: 1}, 'W': {c1: 1}})
        terms.append({'U': {a1: 1}, 'V': {b11: 1}, 'W': {c1: 1}})

    # Define indices
    # A (3x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    
    # B (4x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15
    
    # C (3x4) stored as k*3+i
    # i=0: 0, 3, 6, 9
    # i=1: 1, 4, 7, 10
    # i=2: 2, 5, 8, 11
    
    # Block 1: Top-Left (A_top, B_left) -> C_top_left
    # A_top: [0, 1, 2, 3]
    # B_left: Cols 0,1 -> Rows 0-1: [0,1, 4,5], Rows 2-3: [8,9, 12,13]
    # C_top_left: i=0, k=0,1 -> [0, 3]
    
    # Split A_top into A_t1 [0,1] and A_t2 [2,3]
    # Split B_left into B_tl (Rows 0-1, Cols 0-1) and B_bl (Rows 2-3, Cols 0-1)
    # B_tl: [0,1, 4,5]
    # B_bl: [8,9, 12,13]
    
    # C_top_left is 1x2.
    # We treat it as (1x2) * (2x2) + (1x2) * (2x2)
    # A_t1 * B_tl -> c0, c1 part 1
    # A_t2 * B_bl -> c0, c1 part 2
    
    # Schoolbook for A_t1 * B_tl
    # u: [0, 1]
    # v: [0, 1, 4, 5]
    # w: [0, 3]
    add_schoolbook_1x2x2([0, 1], [0, 1, 4, 5], [0, 3])
    
    # Schoolbook for A_t2 * B_bl
    # u: [2, 3]
    # v: [8, 9, 12, 13]
    # w: [0, 3]
    add_schoolbook_1x2x2([2, 3], [8, 9, 12, 13], [0, 3])
    
    # Block 2: Top-Right (A_top, B_right) -> C_top_right
    # A_top: [0, 1, 2, 3]
    # B_right: Cols 2,3 -> Rows 0-1: [2,3, 6,7], Rows 2-3: [10,11, 14,15]
    # C_top_right: i=0, k=2,3 -> [6, 9]
    
    # A_t1 * B_tr
    # u: [0, 1]
    # v: [2, 3, 6, 7]
    # w: [6, 9]
    add_schoolbook_1x2x2([0, 1], [2, 3, 6, 7], [6, 9])
    
    # A_t2 * B_br
    # u: [2, 3]
    # v: [10, 11, 14, 15]
    # w: [6, 9]
    add_schoolbook_1x2x2([2, 3], [10, 11, 14, 15], [6, 9])
    
    # Block 3: Bottom-Left (A_bot, B_left) -> C_bot_left
    # A_bot: Rows 1,2 -> [4,5,6,7, 8,9,10,11]
    # Split A_bot into A_b1 (Cols 0,1) and A_b2 (Cols 2,3)
    # A_b1: [4,5, 8,9]
    # A_b2: [6,7, 10,11]
    
    # B_left: Rows 0-3, Cols 0-1
    # Split B_left into B_tl (Rows 0-1) and B_bl (Rows 2-3)
    # B_tl: [0,1, 4,5]
    # B_bl: [8,9, 12,13]
    
    # C_bot_left: i=1,2; k=0,1 -> [1,4, 2,5]
    # z00=1, z01=4, z10=2, z11=5
    
    # A_b1 * B_tl
    # u: [4,5, 8,9]
    # v: [0,1, 4,5]
    # w: [1,4, 2,5]
    add_strassen([4,5, 8,9], [0,1, 4,5], [1,4, 2,5])
    
    # A_b2 * B_bl
    # u: [6,7, 10,11]
    # v: [8,9, 12,13]
    # w: [1,4, 2,5]
    add_strassen([6,7, 10,11], [8,9, 12,13], [1,4, 2,5])
    
    # Block 4: Bottom-Right (A_bot, B_right) -> C_bot_right
    # A_b1: [4,5, 8,9]
    # A_b2: [6,7, 10,11]
    
    # B_right: Rows 0-3, Cols 2-3
    # B_tr: [2,3, 6,7]
    # B_br: [10,11, 14,15]
    
    # C_bot_right: i=1,2; k=2,3 -> [7,10, 8,11]
    # z00=7, z01=10, z10=8, z11=11
    
    # A_b1 * B_tr
    # u: [4,5, 8,9]
    # v: [2,3, 6,7]
    # w: [7,10, 8,11]
    add_strassen([4,5, 8,9], [2,3, 6,7], [7,10, 8,11])
    
    # A_b2 * B_br
    # u: [6,7, 10,11]
    # v: [10,11, 14,15]
    # w: [7,10, 8,11]
    add_strassen([6,7, 10,11], [10,11, 14,15], [7,10, 8,11])
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, coeff in term['U'].items():
            U[idx][r] = coeff
        for idx, coeff in term['V'].items():
            V[idx][r] = coeff
        for idx, coeff in term['W'].items():
            W[idx][r] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
