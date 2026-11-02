import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 81
    rank = 81
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    current_rank = 0

    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal current_rank
        if coeff == 0:
            return
        if current_rank >= rank:
             # Should not happen if rank calculation is correct
             return
        U[u_idx][current_rank] += coeff
        V[v_idx][current_rank] += coeff
        W[w_idx][current_rank] += coeff
        current_rank += 1

    # Block Index Definitions
    # A blocks (2x2 each)
    # A00: rows 0-1, cols 0-1 of A (4x4) -> global indices
    A_blocks = {
        '00': [[0, 1], [4, 5]],
        '01': [[2, 3], [6, 7]],
        '10': [[8, 9], [12, 13]],
        '11': [[10, 11], [14, 15]]
    }
    
    # B blocks (2x2 each) for B_L (4x4)
    # B rows 0-3, cols 0-3
    B_blocks_L = {
        '00': [[0, 1], [6, 7]],
        '01': [[2, 3], [8, 9]],
        '10': [[12, 13], [18, 19]],
        '11': [[14, 15], [20, 21]]
    }
    
    # C blocks (2x2 each) for C (6x4)
    # C indices W[k*n + i]
    # C00: k in 0-1, i in 0-1
    C_blocks = {
        '00': [[0, 1], [4, 5]],
        '01': [[2, 3], [6, 7]],
        '10': [[8, 9], [12, 13]],
        '11': [[10, 11], [14, 15]],
        '20': [[16, 17], [20, 21]], # k=4,5; i=0,1
        '21': [[18, 19], [22, 23]]  # k=4,5; i=2,3
    }

    def get_block_indices(block_type, block_id):
        if block_type == 'A':
            return A_blocks[block_id]
        elif block_type == 'B':
            return B_blocks_L[block_id]
        elif block_type == 'C':
            return C_blocks[block_id]

    # Function to add a 2x2 product using Strassen
    # A_comb: list of (block_id, coeff)
    # B_comb: list of (block_id, coeff)
    # C_targets: list of (C_block_id, coeff)
    def add_product_2x2(A_comb, B_comb, C_targets):
        # Construct X and Y as 2x2 matrices of linear forms
        # X[r][c] = list of (u_idx, coeff)
        X = [[[] for _ in range(2)] for _ in range(2)]
        Y = [[[] for _ in range(2)] for _ in range(2)]
        
        for block_id, coeff in A_comb:
            indices = A_blocks[block_id]
            for r in range(2):
                for c in range(2):
                    u_idx = indices[r][c]
                    X[r][c].append((u_idx, coeff))
        
        for block_id, coeff in B_comb:
            indices = B_blocks_L[block_id]
            for r in range(2):
                for c in range(2):
                    v_idx = indices[r][c]
                    Y[r][c].append((v_idx, coeff))
                    
        # Helper to add a linear form (list of (idx, coeff)) to a dictionary
        def add_to_dict(d, idx, coeff):
            if idx in d:
                d[idx] += coeff
            else:
                d[idx] = coeff
                
        # Strassen 7 steps
        # S1 = (X00+X11)(Y00+Y11)
        # S2 = (X10+X11)Y00
        # S3 = X00(Y01-Y11)
        # S4 = X11(Y10-Y00)
        # S5 = (X00+X01)Y11
        # S6 = (X10-X00)(Y00+Y01)
        # S7 = (X01-X11)(Y10+Y11)
        
        # We will define the 7 linear forms for A and B, and which Z entries they affect.
        # Each entry is (A_form, B_form, [(z_r, z_c, sign), ...])
        
        # To avoid creating large lists, we just process each S_j immediately.
        
        def process_s(A_terms, B_terms, Z_targets):
            # A_terms: list of (u_idx, coeff)
            # B_terms: list of (v_idx, coeff)
            # Z_targets: list of (r, c, sign)
            
            # Combine A_terms into a dict
            A_dict = {}
            for idx, c in A_terms:
                add_to_dict(A_dict, idx, c)
                
            B_dict = {}
            for idx, c in B_terms:
                add_to_dict(B_dict, idx, c)
                
            for (r, c, sign) in Z_targets:
                # This S_j contributes to Z[r][c] with 'sign'
                # And Z[r][c] maps to C targets
                for C_block_id, C_coeff in C_targets:
                    w_indices = C_blocks[C_block_id]
                    w_idx = w_indices[r][c]
                    total_coeff = sign * C_coeff
                    
                    # Add rank-1 terms for all pairs in A_dict and B_dict
                    for u_idx, c_u in A_dict.items():
                        for v_idx, c_v in B_dict.items():
                            final_coeff = c_u * c_v * total_coeff
                            if final_coeff != 0:
                                add_term(u_idx, v_idx, w_idx, final_coeff)

        # S1
        A1 = X[0][0] + X[1][1]
        B1 = Y[0][0] + Y[1][1]
        # S1 contributes to Z00 (+1) and Z11 (+1)
        process_s(A1, B1, [(0, 0, 1), (1, 1, 1)])
        
        # S2
        A2 = X[1][0] + X[1][1]
        B2 = Y[0][0]
        # S2 contributes to Z10 (+1) and Z11 (-1)
        process_s(A2, B2, [(1, 0, 1), (1, 1, -1)])
        
        # S3
        A3 = X[0][0]
        B3 = [(idx, coeff) for idx, coeff in Y[0][1]] + [(idx, -coeff) for idx, coeff in Y[1][1]]
        # S3 contributes to Z01 (+1) and Z11 (+1)
        process_s(A3, B3, [(0, 1, 1), (1, 1, 1)])
        
        # S4
        A4 = X[1][1]
        B4 = [(idx, coeff) for idx, coeff in Y[1][0]] + [(idx, -coeff) for idx, coeff in Y[0][0]]
        # S4 contributes to Z00 (+1) and Z10 (+1)
        process_s(A4, B4, [(0, 0, 1), (1, 0, 1)])
        
        # S5
        A5 = X[0][0] + X[0][1]
        B5 = Y[1][1]
        # S5 contributes to Z00 (-1) and Z01 (+1)
        process_s(A5, B5, [(0, 0, -1), (0, 1, 1)])
        
        # S6
        A6 = [(idx, coeff) for idx, coeff in X[1][0]] + [(idx, -coeff) for idx, coeff in X[0][0]]
        B6 = Y[0][0] + Y[0][1]
        # S6 contributes to Z11 (+1)
        process_s(A6, B6, [(1, 1, 1)])
        
        # S7
        A7 = [(idx, coeff) for idx, coeff in X[0][1]] + [(idx, -coeff) for idx, coeff in X[1][1]]
        B7 = Y[1][0] + Y[1][1]
        # S7 contributes to Z00 (+1)
        process_s(A7, B7, [(0, 0, 1)])

    # Define the 7 M's
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    add_product_2x2([('00', 1), ('11', 1)], [('00', 1), ('11', 1)], [('00', 1), ('11', 1)])
    
    # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
    add_product_2x2([('10', 1), ('11', 1)], [('00', 1)], [('10', 1), ('11', -1)])
    
    # M3 = A00(B01-B11) -> C01(+1), C11(+1)
    add_product_2x2([('00', 1)], [('01', 1), ('11', -1)], [('01', 1), ('11', 1)])
    
    # M4 = A11(B10-B00) -> C00(+1), C10(+1)
    add_product_2x2([('11', 1)], [('10', 1), ('00', -1)], [('00', 1), ('10', 1)])
    
    # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
    add_product_2x2([('00', 1), ('01', 1)], [('11', 1)], [('00', -1), ('01', 1)])
    
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    add_product_2x2([('10', 1), ('00', -1)], [('00', 1), ('01', 1)], [('11', 1)])
    
    # M7 = (A01-A11)(B10+B11) -> C00(+1)
    add_product_2x2([('01', 1), ('11', -1)], [('10', 1), ('11', 1)], [('00', 1)])

    # Schoolbook for B_R (cols 4,5)
    # A is 4x4, B_R is 4x2
    # C is 6x4. Rows 4,5 of C are affected.
    # C[k][i] = sum_j A[i][j] B[j][k]
    # k in {4, 5}, i in {0, 1, 2, 3}
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            for k in [4, 5]:
                v_idx = j * 6 + k
                w_idx = k * 4 + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    return {"U": U, "V": V, "W": W}
