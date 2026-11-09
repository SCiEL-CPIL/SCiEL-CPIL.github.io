"""Rewritten program for 4x5 * 5x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Block decomposition strategy:
    # Split A (4x5) into top (2x5) and bottom (2x5)
    # Split B (5x6) into left (5x3) and right (5x3)
    # This gives 4 C-blocks, each 2x3
    #
    # For each 2x3 C-block from A_sub(2x5) * B_sub(5x3):
    # Split middle dimension 5 = 2 + 3
    # C-block = A_L(2x2)*B_T(2x3) + A_R(2x3)*B_B(3x3)
    #
    # For A_L(2x2)*B_T(2x3):
    #   Split B_T into B_TL(2x2) + B_TR(2x1)
    #   Use Strassen for (2x2)(2x2): rank 7
    #   Use schoolbook for (2x2)(2x1): rank 4
    #   Subtotal: 11
    #
    # For A_R(2x3)*B_B(3x3):
    #   Split A_R into A_RL(2x2) + A_RR(2x1)
    #   Split B_B into B_BL(2x3) + B_BR(1x3)
    #   Term A_RL(2x2)*B_BL(2x3): Rank 11 (same as above)
    #   Term A_RR(2x1)*B_BR(1x3): Rank 1 (Outer product)
    #   Subtotal: 12
    #
    # Total per C-block: 11 + 12 = 23
    # Total: 4 * 23 = 92
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term."""
        u_global = [0.0] * (n * m)
        v_global = [0.0] * (m * p)
        w_global = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_global[idx] += val
        for idx, val in v_dict.items():
            v_global[idx] += val
        for idx, val in w_dict.items():
            w_global[idx] += val
        U_list.append(u_global)
        V_list.append(v_global)
        W_list.append(w_global)
    
    def strassen_2x2_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen decomposition for a 2x2 * 2x2 sub-problem.
        """
        a00 = a_row_offset * m + 0
        a01 = a_row_offset * m + 1
        a10 = (a_row_offset + 1) * m + 0
        a11 = (a_row_offset + 1) * m + 1
        
        b00 = 0 * p + b_col_offset
        b01 = 0 * p + (b_col_offset + 1)
        b10 = 1 * p + b_col_offset
        b11 = 1 * p + (b_col_offset + 1)
        
        c00 = c_col_offset * n + c_row_offset
        c01 = (c_col_offset + 1) * n + c_row_offset
        c10 = c_col_offset * n + (c_row_offset + 1)
        c11 = (c_col_offset + 1) * n + (c_row_offset + 1)
        
        # M1: (A11+A22)(B11+B22)
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2: (A11+A21)B11
        add_term(
            {a00: 1, a10: 1},
            {b00: 1},
            {c10: 1}
        )
        # M3: A11(B12-B22)
        add_term(
            {a00: 1},
            {b01: 1, b11: -1},
            {c01: 1}
        )
        # M4: A22(B21-B11)
        add_term(
            {a11: 1},
            {b10: 1, b00: -1},
            {c00: 1, c10: 1}
        )
        # M5: (A12+A22)B22
        add_term(
            {a01: 1, a11: 1},
            {b11: 1},
            {c01: 1, c11: -1}
        )
        # M6: (A11-A21)(B21+B22)
        add_term(
            {a00: 1, a10: -1},
            {b10: 1, b11: 1},
            {c11: 1}
        )
        # M7: (A12-A22)(B11+B21)
        add_term(
            {a01: 1, a11: -1},
            {b00: 1, b10: 1},
            {c00: 1}
        )
    
    def schoolbook_2x2_2x1(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Schoolbook for 2x2 * 2x1 sub-problem.
        """
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    a_idx = (a_row_offset + i) * m + j
                    b_idx = j * p + (b_col_offset + k)
                    c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )
    
    def outer_product_2x1_1x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Outer product for 2x1 * 1x3 sub-problem.
        A is 2x1, B is 1x3.
        """
        # A indices: rows a_row_offset, a_row_offset+1; col 0 (relative to 2x1 block, but we need global col)
        # In the context of A_R (2x3), the 2x1 part is the 3rd column (index 4 globally).
        # So a_idx is (a_row_offset + i) * m + 4
        # B indices: row 0 (relative to 1x3 block, but we need global row)
        # In the context of B_B (3x3), the 1x3 part is the 3rd row (index 4 globally).
        # So b_idx is 4 * p + (b_col_offset + k)
        # C indices: (c_col_offset + k) * n + (c_row_offset + i)
        
        a_col_global = 4
        b_row_global = 4
        
        for i in range(2):
            for k in range(3):
                a_idx = (a_row_offset + i) * m + a_col_global
                b_idx = b_row_global * p + (b_col_offset + k)
                c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                add_term(
                    {a_idx: 1},
                    {b_idx: 1},
                    {c_idx: 1}
                )
    
    def decompose_2x3_3x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Decompose 2x3 * 3x3 sub-problem.
        A is 2x3 (cols 2,3,4 globally).
        B is 3x3 (rows 2,3,4 globally).
        """
        # Part 1: A_L (2x2, cols 2,3) * B_L (2x3, rows 2,3)
        # This is a 2x2 * 2x3 problem.
        # We can reuse the logic: Strassen for 2x2*2x2 and Schoolbook for 2x2*2x1.
        
        # Strassen for 2x2 * 2x2 part (cols 2,3 of A; rows 2,3 of B; cols 0,1 of C)
        # Note: The offsets for strassen_2x2_2x2 assume 0-based local indices.
        # We need to adapt or write a specific function.
        
        # Let's write a specific helper for 2x2 * 2x3 using global offsets.
        def strassen_2x2_2x2_custom(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
            a00 = (a_row_off + 0) * m + (a_col_off + 0)
            a01 = (a_row_off + 0) * m + (a_col_off + 1)
            a10 = (a_row_off + 1) * m + (a_col_off + 0)
            a11 = (a_row_off + 1) * m + (a_col_off + 1)
            
            b00 = (b_row_off + 0) * p + (b_col_off + 0)
            b01 = (b_row_off + 0) * p + (b_col_off + 1)
            b10 = (b_row_off + 1) * p + (b_col_off + 0)
            b11 = (b_row_off + 1) * p + (b_col_off + 1)
            
            c00 = (c_col_off + 0) * n + (c_row_off + 0)
            c01 = (c_col_off + 1) * n + (c_row_off + 0)
            c10 = (c_col_off + 0) * n + (c_row_off + 1)
            c11 = (c_col_off + 1) * n + (c_row_off + 1)
            
            add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
            add_term({a00: 1, a10: 1}, {b00: 1}, {c10: 1})
            add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1})
            add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
            add_term({a01: 1, a11: 1}, {b11: 1}, {c01: 1, c11: -1})
            add_term({a00: 1, a10: -1}, {b10: 1, b11: 1}, {c11: 1})
            add_term({a01: 1, a11: -1}, {b00: 1, b10: 1}, {c00: 1})

        # Schoolbook for 2x2 * 2x1 part (cols 2,3 of A; row 2,3 of B? No, B_L is 2x3.
        # We split B_L (2x3) into B_LL (2x2) and B_LR (2x1).
        # So we need Strassen for A(2x2)*B_LL(2x2) and Schoolbook for A(2x2)*B_LR(2x1).
        
        # A is at cols 2,3. B_LL is at rows 2,3, cols 0,1.
        strassen_2x2_2x2_custom(a_row_offset, 2, 2, b_col_offset, c_row_offset, c_col_offset)
        
        # A is at cols 2,3. B_LR is at rows 2,3, col 2.
        # Schoolbook 2x2 * 2x1
        def schoolbook_2x2_2x1_custom(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
            for i in range(2):
                for j in range(2):
                    for k in range(1):
                        a_idx = (a_row_off + i) * m + (a_col_off + j)
                        b_idx = (b_row_off + j) * p + (b_col_off + k)
                        c_idx = (c_col_off + k) * n + (c_row_off + i)
                        add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})
        
        schoolbook_2x2_2x1_custom(a_row_offset, 2, 2, b_col_offset + 2, c_row_offset, c_col_offset + 2)
        
        # Part 2: A_R (2x1, col 4) * B_R (1x3, row 4)
        outer_product_2x1_1x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset)
        
    # Build the decomposition
    for a_row in [0, 2]:
        for b_col in [0, 3]:
            c_row = a_row
            c_col = b_col
            
            # Part 1: A_sub_L(2x2) * B_sub_T(2x3)
            # A is cols 0,1. B is rows 0,1.
            strassen_2x2_2x2(a_row, b_col, c_row, c_col)
            schoolbook_2x2_2x1(a_row, b_col + 2, c_row, c_col + 2)
            
            # Part 2: A_sub_R(2x3) * B_sub_B(3x3)
            decompose_2x3_3x3(a_row, b_col, c_row, c_col)
    
    # Build final matrices
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_list[c][i]
        for j in range(m * p):
            V[j][c] = V_list[c][j]
        for k in range(p * n):
            W[k][c] = W_list[c][k]
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return best_result
