import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 105
    # 49 for 4x4x4 Strassen
    # 32 for 4x4x2 Schoolbook
    # 24 for 4x1x6 Schoolbook
    
    U = [[0] * 105 for _ in range(n * m)]
    V = [[0] * 105 for _ in range(m * p)]
    W = [[0] * 105 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx][col] += coeff
        col += 1

    # 1. Strassen for 4x4x4
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    # B blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # Define block indices
    def get_a_block_indices(block_id):
        # block_id: 11, 12, 21, 22
        r1 = 0 if block_id[0] == '1' else 2
        c1 = 0 if block_id[1] == '1' else 2
        return [(r1+i, c1+j) for i in range(2) for j in range(2)]
        
    def get_b_block_indices(block_id):
        r1 = 0 if block_id[0] == '1' else 2
        c1 = 0 if block_id[1] == '1' else 2
        return [(r1+i, c1+j) for i in range(2) for j in range(2)]
        
    def get_c_block_indices(block_id):
        r1 = 0 if block_id[0] == '1' else 2
        c1 = 0 if block_id[1] == '1' else 2
        return [(r1+i, c1+j) for i in range(2) for j in range(2)]

    # Strassen steps for 4x4
    # Each step defines X (linear combo of A blocks) and Y (linear combo of B blocks)
    # And which C blocks it contributes to with what sign
    strassen_4x4_steps = [
        # (X_blocks, Y_blocks, C_blocks_with_signs)
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}, {'11': 1, '22': 1}), # M1
        ({'21': 1, '22': 1}, {'11': 1}, {'21': 1, '22': -1}), # M2
        ({'11': 1}, {'21': 1, '22': -1}, {'12': 1, '22': 1}), # M3
        ({'22': 1}, {'11': 1, '21': -1}, {'11': 1, '21': 1}), # M4
        ({'11': 1, '12': 1}, {'22': 1}, {'11': -1, '12': 1}), # M5
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}, {'22': 1}), # M6
        ({'12': 1, '22': -1}, {'21': 1, '22': 1}, {'11': 1}) # M7
    ]
    
    # Strassen steps for 2x2
    # X is 2x2, Y is 2x2
    # X entries: X00, X01, X10, X11
    # Y entries: Y00, Y01, Y10, Y11
    # P1 = (X00+X11)(Y00+Y11) -> C00(+), C11(+)
    # P2 = (X10+X11)Y00 -> C10(+), C11(-)
    # P3 = X00(Y10-Y11) -> C01(+), C11(+)
    # P4 = X11(Y00-Y10) -> C00(+), C10(+)
    # P5 = (X00+X01)Y11 -> C00(-), C01(+)
    # P6 = (X10-X00)(Y00+Y01) -> C11(+)
    # P7 = (X01-X11)(Y10+Y11) -> C00(+)
    
    strassen_2x2_steps = [
        # (X_coeffs, Y_coeffs, C_coeffs)
        # X_coeffs: {00: c, 01: c, 10: c, 11: c}
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}), # P1
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}), # P2
        ({'00': 1}, {'10': 1, '11': -1}, {'01': 1, '11': 1}), # P3
        ({'11': 1}, {'00': 1, '10': -1}, {'00': 1, '10': 1}), # P4
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}), # P5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}), # P6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}) # P7
    ]
    
    for step4 in strassen_4x4_steps:
        x_blocks, y_blocks, c_blocks = step4
        for step2 in strassen_2x2_steps:
            x_entry_coeffs, y_entry_coeffs, c_entry_coeffs = step2
            
            # Construct U coeffs (linear combo of A entries)
            u_coeffs = {}
            for b_id, b_coeff in x_blocks.items():
                a_indices = get_a_block_indices(b_id)
                # Map X entry (r,c) to A block entry
                # X is 2x2. Entry (r,c) corresponds to row r, col c of the block.
                for r in range(2):
                    for c in range(2):
                        entry_id = f"{r}{c}"
                        if entry_id in x_entry_coeffs:
                            e_coeff = x_entry_coeffs[entry_id]
                            total_coeff = b_coeff * e_coeff
                            # Find the global A index
                            # a_indices is list of (row, col)
                            # The order in get_a_block_indices is row-major 2x2
                            # (0,0), (0,1), (1,0), (1,1)
                            # So index r*2+c in the list?
                            # No, get_a_block_indices returns list of tuples.
                            # Let's re-verify the mapping.
                            # X entry (r,c) is the element at row r, col c of the 2x2 matrix X.
                            # X is formed from A blocks.
                            # The element (r,c) of X comes from the element (r,c) of each A block in the sum.
                            # So for block b_id, the element is at global index (block_row + r, block_col + c).
                            # get_a_block_indices returns list of (row, col) for the block.
                            # We need the specific element (r,c).
                            # Let's just calculate the global index directly.
                            br1 = 0 if b_id[0] == '1' else 2
                            bc1 = 0 if b_id[1] == '1' else 2
                            global_row = br1 + r
                            global_col = bc1 + c
                            a_idx = global_row * m + global_col
                            u_coeffs[a_idx] = u_coeffs.get(a_idx, 0) + total_coeff
            
            # Construct V coeffs (linear combo of B entries)
            v_coeffs = {}
            for b_id, b_coeff in y_blocks.items():
                for r in range(2):
                    for c in range(2):
                        entry_id = f"{r}{c}"
                        if entry_id in y_entry_coeffs:
                            e_coeff = y_entry_coeffs[entry_id]
                            total_coeff = b_coeff * e_coeff
                            br1 = 0 if b_id[0] == '1' else 2
                            bc1 = 0 if b_id[1] == '1' else 2
                            global_row = br1 + r
                            global_col = bc1 + c
                            b_idx = global_row * p + global_col
                            v_coeffs[b_idx] = v_coeffs.get(b_idx, 0) + total_coeff
                            
            # Construct W coeffs (linear combo of C entries)
            w_coeffs = {}
            for c_id, c_coeff in c_blocks.items():
                for r in range(2):
                    for c in range(2):
                        entry_id = f"{r}{c}"
                        if entry_id in c_entry_coeffs:
                            e_coeff = c_entry_coeffs[entry_id]
                            total_coeff = c_coeff * e_coeff
                            cr1 = 0 if c_id[0] == '1' else 2
                            cc1 = 0 if c_id[1] == '1' else 2
                            global_row = cr1 + r
                            global_col = cc1 + c
                            # W index is k*n+i where k is row, i is col
                            w_idx = global_row * n + global_col
                            w_coeffs[w_idx] = w_coeffs.get(w_idx, 0) + total_coeff
                            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # 2. Schoolbook for 4x4x2
    # A: rows 0-3, cols 0-3
    # B: rows 0-3, cols 4-5
    # C: rows 4-5, cols 0-3
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_coeffs = {i * m + j: 1}
                v_coeffs = {j * p + k: 1}
                w_coeffs = {k * n + i: 1}
                add_term(u_coeffs, v_coeffs, w_coeffs)
                
    # 3. Schoolbook for 4x1x6
    # A: rows 0-3, col 4
    # B: row 4, cols 0-5
    # C: rows 0-5, cols 0-3
    for i in range(4):
        for k in range(6):
            u_coeffs = {i * m + 4: 1}
            v_coeffs = {4 * p + k: 1}
            w_coeffs = {k * n + i: 1}
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    return {"U": U, "V": V, "W": W}

