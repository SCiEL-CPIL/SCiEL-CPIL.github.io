import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4x4 Strassen (with schoolbook 2x2 blocks): 7 * 8 = 56
    # A01 B10 (4x1 * 1x4): 16
    # A00 B01 (4x4 * 4x1): 16
    # A01 B11 (4x1 * 1x1): 4
    # A10 B00 (1x4 * 4x4): 16
    # A11 B10 (1x1 * 1x4): 4
    # A10 B01 (1x4 * 4x1): 4
    # A11 B11 (1x1 * 1x1): 1
    # Total: 56 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 117
    
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # 1. A00 B00 (4x4) using Strassen
    # Strassen 2x2 formulas (0-indexed):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We decompose each Mk (2x2 mult) using schoolbook (8 terms).
    # Schoolbook 2x2 terms:
    # T1: A00 B00 -> C00
    # T2: A01 B10 -> C00
    # T3: A00 B01 -> C01
    # T4: A01 B11 -> C01
    # T5: A10 B00 -> C10
    # T6: A11 B10 -> C10
    # T7: A10 B01 -> C11
    # T8: A11 B11 -> C11
    
    # For each Mk, we define the coefficients for A blocks (a00, a01, a10, a11) and B blocks (b00, b01, b10, b11).
    # Then for each schoolbook term (t_a, t_b, t_c), we add the product of coefficients.
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
        """
        a_coeffs: dict {block_idx: coeff} for A blocks in the 2x2 multiplication
        b_coeffs: dict {block_idx: coeff} for B blocks
        c_coeffs: dict {block_idx: coeff} for C blocks (result of this 2x2 mult)
        """
        # Schoolbook 2x2 terms
        # (a_idx, b_idx, c_idx)
        terms = [
            (0, 0, 0), # A00 B00 -> C00
            (1, 2, 0), # A01 B10 -> C00
            (0, 1, 1), # A00 B01 -> C01
            (1, 3, 1), # A01 B11 -> C01
            (2, 0, 2), # A10 B00 -> C10
            (3, 2, 2), # A11 B10 -> C10
            (2, 1, 3), # A10 B01 -> C11
            (3, 3, 3)  # A11 B11 -> C11
        ]
        
        for t_a, t_b, t_c in terms:
            ca = a_coeffs.get(t_a, 0)
            cb = b_coeffs.get(t_b, 0)
            cc = c_coeffs.get(t_c, 0)
            
            if ca == 0 or cb == 0 or cc == 0:
                continue
                
            coeff = ca * cb * cc
            
            # Map 2x2 block indices to 4x4 indices
            # A blocks in 4x4:
            # 0: A00 (rows 0,1; cols 0,1)
            # 1: A01 (rows 0,1; cols 2,3)
            # 2: A10 (rows 2,3; cols 0,1)
            # 3: A11 (rows 2,3; cols 2,3)
            
            # B blocks in 4x4:
            # 0: B00 (rows 0,1; cols 0,1)
            # 1: B01 (rows 0,1; cols 2,3)
            # 2: B10 (rows 2,3; cols 0,1)
            # 3: B11 (rows 2,3; cols 2,3)
            
            # C blocks in 4x4:
            # 0: C00 (rows 0,1; cols 0,1)
            # 1: C01 (rows 0,1; cols 2,3)
            # 2: C10 (rows 2,3; cols 0,1)
            # 3: C11 (rows 2,3; cols 2,3)
            
            # For a specific schoolbook term t_a, t_b, t_c:
            # We need to iterate over the 2x2 indices within the blocks.
            # Let i, j, k be indices in 0..1 for the 2x2 blocks.
            # A index in 4x4: (block_row * 2 + i) * 4 + (block_col * 2 + j)
            # B index in 4x4: (block_row * 2 + j) * 4 + (block_col * 2 + k)
            # C index in 4x4: (block_row * 2 + i) * 4 + (block_col * 2 + k)
            
            # Block row/col for t_a:
            # t_a is 0..3.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            a_br, a_bc = divmod(t_a, 2)
            # t_b
            b_br, b_bc = divmod(t_b, 2)
            # t_c
            c_br, c_bc = divmod(t_c, 2)
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # A index
                        a_r = a_br * 2 + i
                        a_c = a_bc * 2 + j
                        a_idx = a_r * 4 + a_c
                        
                        # B index
                        b_r = b_br * 2 + j
                        b_c = b_bc * 2 + k
                        b_idx = b_r * 4 + b_c
                        
                        # C index
                        c_r = c_br * 2 + i
                        c_c = c_bc * 2 + k
                        c_idx = c_r * 4 + c_c
                        
                        # Map to 5x5 indices
                        # A is 5x5. 4x4 part is top-left.
                        # U index: i_5 * 5 + j_5
                        u_5_idx = a_r * 5 + a_c
                        # V index: j_5 * 5 + k_5
                        v_5_idx = b_r * 5 + b_c
                        # W index: k_5 * 5 + i_5
                        w_5_idx = c_c * 5 + c_r
                        
                        add_term(u_5_idx, v_5_idx, w_5_idx, coeff)

    # Define the 7 Strassen terms
    # M1 = (A00+A11)(B00+B11)
    add_strassen_term({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    
    # M2 = (A10+A11)B00
    add_strassen_term({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}) # Wait, C indices for M2?
    # M2 contributes to C10 and C11.
    # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So M2 is a 2x2 matrix.
    # Its C00 goes to C10? No.
    # M2 is a 2x2 matrix multiplication result.
    # Let's call the result matrix R2.
    # C10 (4x4 block) gets R2?
    # No, C10 is 4x4? No, C10 in 4x4 is 2x2.
    # In 4x4 matrix C, C10 is the bottom-left 2x2 block.
    # So M2 contributes to C10.
    # But M2 is a 2x2 matrix.
    # So C10 = M2 + M4.
    # This means the entire 2x2 matrix M2 is added to the 2x2 matrix C10.
    # So c_coeffs for M2 should be {2: 1} (block 2 is C10).
    # Wait, M2 is a 2x2 matrix.
    # The "c_coeffs" in my helper function maps the 2x2 result blocks to the 4x4 C blocks.
    # If M2 is added to C10, then all 4 entries of M2 go to C10.
    # So c_coeffs = {0: 1, 1: 1, 2: 1, 3: 1}? No.
    # The helper function adds to C blocks based on the schoolbook term's c_idx.
    # If I want M2 to go to C10 (block 2), I should map all c_idx to block 2?
    # No, that's wrong.
    # M2 is a 2x2 matrix.
    # C10 is a 2x2 matrix.
    # C10 = M2 + M4.
    # This implies element-wise addition.
    # So M2[0,0] goes to C10[0,0].
    # In my helper, t_c determines which part of the 2x2 result goes where.
    # If I want the entire 2x2 result M2 to go to C10, I need to map:
    # t_c=0 (R[0,0]) -> C10[0,0] (block 2, idx 0,0)
    # t_c=1 (R[0,1]) -> C10[0,1] (block 2, idx 0,1)
    # t_c=2 (R[1,0]) -> C10[1,0] (block 2, idx 1,0)
    # t_c=3 (R[1,1]) -> C10[1,1] (block 2, idx 1,1)
    
    # My helper function currently maps t_c to a block.
    # If I pass c_coeffs = {0: 1, 1: 1, 2: 1, 3: 1}, it will add to all blocks?
    # No, the helper iterates t_c in 0..3.
    # If c_coeffs has {0: 1, 1: 1, 2: 1, 3: 1}, it will add to block 0, 1, 2, 3?
    # No, c_coeffs is a dict.
    # If I want M2 to go to C10 (block 2), I should set c_coeffs = {2: 1}?
    # But then only t_c=2 will be processed?
    # No, the helper uses c_coeffs.get(t_c, 0).
    # If I want all 4 terms of M2 to go to C10, I need to map t_c=0->2, t_c=1->2, t_c=2->2, t_c=3->2.
    # This is not supported by the current helper which maps t_c to a single block index.
    
    # I need to modify the helper to handle mapping of the 2x2 result to the 4x4 C blocks.
    # Actually, the Strassen formulas are linear combinations of the 2x2 blocks.
    # C00 = M1 + M4 - M5 + M7.
    # This means C00 (block 0) is the sum of the 2x2 matrices M1, M4, etc.
    # So for M1, all 4 terms go to C00.
    # So c_coeffs for M1 should be {0: 1, 1: 1, 2: 1, 3: 1}?
    # No, that would add M1 to C00, C01, C10, C11.
    # I need to add M1 to C00 only.
    # So I need to map t_c=0,1,2,3 to block 0.
    
    # Let's rewrite the helper to take a mapping of t_c to block index.
    
    def add_strassen_term_v2(a_coeffs, b_coeffs, c_map):
        """
        a_coeffs: dict {block_idx: coeff}
        b_coeffs: dict {block_idx: coeff}
        c_map: dict {t_c_idx: block_idx}
        """
        terms = [
            (0, 0, 0), (1, 2, 0),
            (0, 1, 1), (1, 3, 1),
            (2, 0, 2), (3, 2, 2),
            (2, 1, 3), (3, 3, 3)
        ]
        
        for t_a, t_b, t_c in terms:
            ca = a_coeffs.get(t_a, 0)
            cb = b_coeffs.get(t_b, 0)
            c_block = c_map.get(t_c, -1)
            
            if ca == 0 or cb == 0 or c_block == -1:
                continue
                
            coeff = ca * cb
            
            a_br, a_bc = divmod(t_a, 2)
            b_br, b_bc = divmod(t_b, 2)
            c_br, c_bc = divmod(c_block, 2)
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        a_r = a_br * 2 + i
                        a_c = a_bc * 2 + j
                        a_idx = a_r * 4 + a_c
                        
                        b_r = b_br * 2 + j
                        b_c = b_bc * 2 + k
                        b_idx = b_r * 4 + b_c
                        
                        c_r = c_br * 2 + i
                        c_c = c_bc * 2 + k
                        c_idx = c_r * 4 + c_c
                        
                        u_5_idx = a_r * 5 + a_c
                        v_5_idx = b_r * 5 + b_c
                        w_5_idx = c_c * 5 + c_r
                        
                        add_term(u_5_idx, v_5_idx, w_5_idx, coeff)

    # M1 -> C00
    add_strassen_term_v2({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 0, 1: 0, 2: 0, 3: 0})
    
    # M2 -> C10
    add_strassen_term_v2({2: 1, 3: 1}, {0: 1}, {0: 2, 1: 2, 2: 2, 3: 2})
    
    # M3 -> C01
    add_strassen_term_v2({0: 1}, {1: 1, 3: -1}, {0: 1, 1: 1, 2: 1, 3: 1})
    
    # M4 -> C00, C10
    # C00 = M1 + M4 ...
    # C10 = M2 + M4 ...
    # So M4 contributes to both.
    # We call it twice?
    # M4 = A11(B01-B00)
    add_strassen_term_v2({3: 1}, {1: 1, 0: -1}, {0: 0, 1: 0, 2: 0, 3: 0}) # To C00
    add_strassen_term_v2({3: 1}, {1: 1, 0: -1}, {0: 2, 1: 2, 2: 2, 3: 2}) # To C10
    
    # M5 -> C00, C01
    # C00 = ... - M5 ...
    # C01 = M3 + M5
    add_strassen_term_v2({0: 1, 1: 1}, {3: 1}, {0: 0, 1: 0, 2: 0, 3: 0}) # To C00, coeff -1?
    # Wait, the coeff is handled by a_coeffs, b_coeffs.
    # I need to pass the sign.
    # Let's just pass the sign in a_coeffs or b_coeffs.
    
    # Redo with signs
    # M1: (A00+A11)(B00+B11) -> C00 (+1)
    add_strassen_term_v2({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 0, 1: 0, 2: 0, 3: 0})
    
    # M2: (A10+A11)B00 -> C10 (+1)
    add_strassen_term_v2({2: 1, 3: 1}, {0: 1}, {0: 2, 1: 2, 2: 2, 3: 2})
    
    # M3: A00(B01-B11) -> C01 (+1)
    add_strassen_term_v2({0: 1}, {1: 1, 3: -1}, {0: 1, 1: 1, 2: 1, 3: 1})
    
    # M4: A11(B01-B00) -> C00 (+1), C10 (+1)
    add_strassen_term_v2({3: 1}, {1: 1, 0: -1}, {0: 0, 1: 0, 2: 0, 3: 0})
    add_strassen_term_v2({3: 1}, {1: 1, 0: -1}, {0: 2, 1: 2, 2: 2, 3: 2})
    
    # M5: (A00+A01)B11 -> C00 (-1), C01 (+1)
    add_strassen_term_v2({0: -1, 1: -1}, {3: 1}, {0: 0, 1: 0, 2: 0, 3: 0})
    add_strassen_term_v2({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1, 2: 1, 3: 1})
    
    # M6: (A10-A00)(B00+B01) -> C11 (+1)
    add_strassen_term_v2({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 3, 1: 3, 2: 3, 3: 3})
    
    # M7: (A11-A01)(B01+B11) -> C00 (+1)
    add_strassen_term_v2({3: 1, 1: -1}, {1: 1, 3: 1}, {0: 0, 1: 0, 2: 0, 3: 0})
    
    # 2. A01 B10 (4x1 * 1x4)
    # i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 5 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # 3. A00 B01 (4x4 * 4x1)
    # i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 5 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # 4. A01 B11 (4x1 * 1x1)
    # i in 0..3
    for i in range(4):
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        add_term(u_idx, v_idx, w_idx, 1)
        
    # 5. A10 B00 (1x4 * 4x4)
    # j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            add_term(u_idx, v_idx, w_idx, 1)
            
    # 6. A11 B10 (1x1 * 1x4)
    # k in 0..3
    for k in range(4):
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + k
        w_idx = k * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1)
        
    # 7. A10 B01 (1x4 * 4x1)
    # j in 0..3
    for j in range(4):
        u_idx = 4 * 5 + j
        v_idx = j * 5 + 4
        w_idx = 4 * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1)
        
    # 8. A11 B11 (1x1 * 1x1)
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    add_term(u_idx, v_idx, w_idx, 1)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
