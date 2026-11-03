import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # U: 16 x R
    # V: 20 x R
    # W: 20 x R
    
    # We will collect terms as a list of (u_dict, v_dict, w_dict)
    # u_dict: {index: coeff}
    # v_dict: {index: coeff}
    # w_dict: {index: coeff}
    
    terms = []
    
    def add_term(u_d, v_d, w_d):
        # Merge if necessary? No, each term is a rank-1 component.
        # But wait, if we have multiple contributions to the same W index from the same L_A, L_B,
        # they can be combined into one term.
        # However, Strassen generates distinct L_A, L_B pairs.
        # So each (L_A, L_B) pair is unique.
        # So we just append.
        terms.append((u_d, v_d, w_d))

    # Helper to create linear form dict
    def make_dict(pairs):
        d = {}
        for idx, coeff in pairs:
            if coeff != 0:
                if idx in d:
                    d[idx] += coeff
                else:
                    d[idx] = coeff
        return d

    # Strassen 2x2x2 decomposition
    # Inputs: A_block (2x2 dict of linear forms), B_block (2x2 dict of linear forms)
    # Output: list of (u_dict, v_dict, w_dict) for the 7 scalar products
    # and the mapping of these products to the 2x2 output block Z.
    
    def strassen_2x2(A_block, B_block):
        # A_block[i][j] is a dict {A_index: coeff}
        # B_block[i][j] is a dict {B_index: coeff}
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = make_dict(list(A_block[0][0].items()) + list(A_block[1][1].items()))
        v1 = make_dict(list(B_block[0][0].items()) + list(B_block[1][1].items()))
        
        # M2 = (A10 + A11)B00
        u2 = make_dict(list(A_block[1][0].items()) + list(A_block[1][1].items()))
        v2 = B_block[0][0]
        
        # M3 = A00(B01 - B11)
        u3 = A_block[0][0]
        v3 = make_dict(list(B_block[0][1].items()) + [(k, -v) for k, v in B_block[1][1].items()])
        
        # M4 = A11(B10 - B00)
        u4 = A_block[1][1]
        v4 = make_dict(list(B_block[1][0].items()) + [(k, -v) for k, v in B_block[0][0].items()])
        
        # M5 = (A00 + A01)B11
        u5 = make_dict(list(A_block[0][0].items()) + list(A_block[0][1].items()))
        v5 = B_block[1][1]
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = make_dict(list(A_block[1][0].items()) + [(k, -v) for k, v in A_block[0][0].items()])
        v6 = make_dict(list(B_block[0][0].items()) + list(B_block[0][1].items()))
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = make_dict(list(A_block[0][1].items()) + [(k, -v) for k, v in A_block[1][1].items()])
        v7 = make_dict(list(B_block[1][0].items()) + list(B_block[1][1].items()))
        
        products = [
            (u1, v1),
            (u2, v2),
            (u3, v3),
            (u4, v4),
            (u5, v5),
            (u6, v6),
            (u7, v7)
        ]
        
        # Coefficients for Z00, Z01, Z10, Z11
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # Z00
            [0, 0, 1, 0, 1, 0, 0],  # Z01
            [0, 1, 0, 1, 0, 0, 0],  # Z10
            [1, -1, 1, 0, 0, 1, 0]  # Z11
        ]
        
        return products, coeffs

    # Build A and B blocks for 4x4x4
    # A_block[I][J] is 2x2 dict of linear forms
    # A[i][j] index = i*4 + j
    
    A_blocks = [[None for _ in range(2)] for _ in range(2)]
    B_blocks = [[None for _ in range(2)] for _ in range(2)]
    
    for I in range(2):
        for J in range(2):
            A_blocks[I][J] = [[None for _ in range(2)] for _ in range(2)]
            for pi in range(2):
                for pj in range(2):
                    i = 2*I + pi
                    j = 2*J + pj
                    idx = i*4 + j
                    A_blocks[I][J][pi][pj] = {idx: 1}
    
    for J in range(2):
        for K in range(2):
            B_blocks[J][K] = [[None for _ in range(2)] for _ in range(2)]
            for pj in range(2):
                for pk in range(2):
                    j = 2*J + pj
                    k = 2*K + pk
                    idx = j*5 + k
                    B_blocks[J][K][pj][pk] = {idx: 1}
    
    # Iterate over the 7 block products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_block_product(A_left, B_right, C_target_blocks):
        # A_left, B_right are 2x2 dicts of dicts
        # C_target_blocks is a list of (I, K, coeff_matrix)
        # coeff_matrix is 2x2 list of coeffs for Z00..Z11 contribution
        
        products, coeffs = strassen_2x2(A_left, B_right)
        
        for pi in range(2):
            for pk in range(2):
                # coeffs[pi][pk] is the vector of coeffs for M1..M7
                for s in range(7):
                    c = coeffs[pi][pk][s]
                    if c == 0: continue
                    u_d, v_d = products[s]
                    # Contribution to C
                    # The product S_s contributes to Z[pi][pk] with coeff c
                    # Z is a block of C.
                    # We need to add this to the global C indices.
                    # But we don't know I, K yet.
                    # This function is called for each specific M_t.
                    # So we need to pass I, K.
                    pass

    # Let's restructure.
    # We define the 7 M_t explicitly.
    
    def process_M(A_left, B_right, I, K):
        products, coeffs = strassen_2x2(A_left, B_right)
        for pi in range(2):
            for pk in range(2):
                for s in range(7):
                    c = coeffs[pi][pk][s]
                    if c == 0: continue
                    u_d, v_d = products[s]
                    # C index
                    i = 2*I + pi
                    k = 2*K + pk
                    w_idx = k*4 + i
                    w_d = {w_idx: c}
                    add_term(u_d, v_d, w_d)

    # M1
    A_left = [[make_dict(list(A_blocks[0][0][pi][pj].items()) + list(A_blocks[1][1][pi][pj].items())) for pj in range(2)] for pi in range(2)]
    B_right = [[make_dict(list(B_blocks[0][0][pi][pj].items()) + list(B_blocks[1][1][pi][pj].items())) for pj in range(2)] for pi in range(2)]
    # Wait, A_left[pi][pj] should be the (pi,pj) element of the block matrix.
    # My A_blocks[I][J] is a 2x2 matrix of dicts.
    # So A_left[pi][pj] = A_blocks[0][0][pi][pj] + A_blocks[1][1][pi][pj]
    
    # Let's make a helper to add block matrices
    def add_block(A1, A2):
        return [[make_dict(list(A1[pi][pj].items()) + list(A2[pi][pj].items())) for pj in range(2)] for pi in range(2)]
    
    def sub_block(A1, A2):
        return [[make_dict(list(A1[pi][pj].items()) + [(k, -v) for k, v in A2[pi][pj].items()]) for pj in range(2)] for pi in range(2)]
    
    # M1 = (A00+A11)(B00+B11)
    process_M(add_block(A_blocks[0][0], A_blocks[1][1]), add_block(B_blocks[0][0], B_blocks[1][1]), 0, 0)
    
    # M2 = (A10+A11)B00
    process_M(add_block(A_blocks[1][0], A_blocks[1][1]), B_blocks[0][0], 1, 0)
    
    # M3 = A00(B01-B11)
    process_M(A_blocks[0][0], sub_block(B_blocks[0][1], B_blocks[1][1]), 0, 1)
    
    # M4 = A11(B10-B00)
    process_M(A_blocks[1][1], sub_block(B_blocks[1][0], B_blocks[0][0]), 1, 1)
    
    # M5 = (A00+A01)B11
    process_M(add_block(A_blocks[0][0], A_blocks[0][1]), B_blocks[1][1], 0, 1)
    
    # M6 = (A10-A00)(B00+B01)
    process_M(sub_block(A_blocks[1][0], A_blocks[0][0]), add_block(B_blocks[0][0], B_blocks[0][1]), 1, 1)
    
    # M7 = (A01-A11)(B10+B11)
    process_M(sub_block(A_blocks[0][1], A_blocks[1][1]), add_block(B_blocks[1][0], B_blocks[1][1]), 0, 0)
    
    # Part 2: 4x1x5
    for i in range(4):
        for k in range(5):
            u_idx = i*4 + 3
            v_idx = 3*5 + k
            w_idx = k*4 + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    # Construct U, V, W
    R = len(terms)
    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(20)]
    W = [[0]*R for _ in range(20)]
    
    for r, (u_d, v_d, w_d) in enumerate(terms):
        for idx, val in u_d.items():
            U[idx][r] = val
        for idx, val in v_d.items():
            V[idx][r] = val
        for idx, val in w_d.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
