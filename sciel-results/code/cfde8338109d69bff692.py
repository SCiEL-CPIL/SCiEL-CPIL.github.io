"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. Keys are (row) indices in global space."""
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
        A rows: a_row_offset, a_row_offset+1
        B cols: b_col_offset, b_col_offset+1
        C rows: c_row_offset, c_row_offset+1
        C cols: c_col_offset, c_col_offset+1
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
        A rows: a_row_offset, a_row_offset+1, cols 0,1
        B rows 0,1, col b_col_offset
        C rows: c_row_offset, c_row_offset+1, col c_col_offset
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
                    
    def schoolbook_2x1_3(a_row_offset, b_row_offset, c_row_offset, c_col_offset):
        """
        Schoolbook for 2x1 * 1x3 sub-problem.
        A rows: a_row_offset, a_row_offset+1, col 0
        B row: b_row_offset, cols 0,1,2
        C rows: c_row_offset, c_row_offset+1, cols c_col_offset to c_col_offset+2
        """
        for i in range(2):
            for k in range(3):
                a_idx = (a_row_offset + i) * m + 0
                b_idx = b_row_offset * p + k
                c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                add_term(
                    {a_idx: 1},
                    {b_idx: 1},
                    {c_idx: 1}
                )
                
    def block_2x3_3(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Decomposition for 2x3 * 3x3.
        A rows: a_row_offset, a_row_offset+1, cols 0,1,2
        B rows 0,1,2, cols b_col_offset to b_col_offset+2
        C rows: c_row_offset, c_row_offset+1, cols c_col_offset to c_col_offset+2
        
        Split A into A1 (2x1) and A2 (2x2)
        Split B into B1 (1x3) and B2 (2x3)
        C = A1*B1 + A2*B2
        
        A1*B1: 2x1 * 1x3 (Rank 6)
        A2*B2: 2x2 * 2x3
             Split B2 into B2L (2x2) and B2R (2x1)
             A2*B2L: 2x2 * 2x2 (Rank 7, Strassen)
             A2*B2R: 2x2 * 2x1 (Rank 4, Schoolbook)
        """
        # Part 1: A1 (2x1) * B1 (1x3)
        # A1 uses A col 0. B1 uses B row 0.
        schoolbook_2x1_3(a_row_offset, 0, c_row_offset, c_col_offset)
        
        # Part 2: A2 (2x2) * B2 (2x3)
        # A2 uses A cols 1, 2. B2 uses B rows 1, 2.
        # We need to map these to local 0,1 for the helper functions.
        
        # Sub-part 2a: A2 * B2L (2x2 * 2x2)
        # A2 cols: 1, 2 (local 0, 1)
        # B2L rows: 1, 2 (local 0, 1), cols 0, 1 (local 0, 1)
        # C cols: 0, 1 (local 0, 1)
        
        # We need a generic strassen that takes explicit indices?
        # The current strassen_2x2_2x2 assumes A cols 0,1 and B rows 0,1.
        # We need to adapt or create a new helper.
        
        # Let's create a helper that takes explicit indices for A cols and B rows.
        # Or we can just write the terms manually here? No, too long.
        # Let's create `strassen_2x2_2x2_generic`
        
        pass # Placeholder, will implement generic version below
        
    # Generic Strassen for 2x2 * 2x2 with arbitrary A cols and B rows
    def strassen_2x2_2x2_generic(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        A: rows a_row_offset, a_row_offset+1; cols a_col_offset, a_col_offset+1
        B: rows b_row_offset, b_row_offset+1; cols b_col_offset, b_col_offset+1
        C: rows c_row_offset, c_row_offset+1; cols c_col_offset, c_col_offset+1
        """
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + (a_col_offset + 1)
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + (a_col_offset + 1)
        
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + (b_col_offset + 1)
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + (b_col_offset + 1)
        
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
        
    def schoolbook_2x2_2x1_generic(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Schoolbook for 2x2 * 2x1.
        A: rows a_row_offset, a_row_offset+1; cols a_col_offset, a_col_offset+1
        B: rows b_row_offset, b_row_offset+1; col b_col_offset
        C: rows c_row_offset, c_row_offset+1; col c_col_offset
        """
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    a_idx = (a_row_offset + i) * m + (a_col_offset + j)
                    b_idx = (b_row_offset + j) * p + b_col_offset
                    c_idx = c_col_offset * n + (c_row_offset + i)
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )
                    
    # Re-implement block_2x3_3 with generic helpers
    def block_2x3_3(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A rows: a_row_offset, a_row_offset+1
        # A cols: 0, 1, 2 (local)
        # B rows: 0, 1, 2 (local)
        # B cols: b_col_offset, b_col_offset+1, b_col_offset+2
        # C rows: c_row_offset, c_row_offset+1
        # C cols: c_col_offset, c_col_offset+1, c_col_offset+2
        
        # Part 1: A1 (2x1) * B1 (1x3)
        # A1: col 0. B1: row 0.
        # C cols: 0, 1, 2
        for i in range(2):
            for k in range(3):
                a_idx = (a_row_offset + i) * m + 0
                b_idx = 0 * p + k
                c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                add_term(
                    {a_idx: 1},
                    {b_idx: 1},
                    {c_idx: 1}
                )
        
        # Part 2: A2 (2x2) * B2 (2x3)
        # A2: cols 1, 2.
        # B2: rows 1, 2.
        
        # Sub-part 2a: A2 * B2L (2x2 * 2x2)
        # A2 cols: 1, 2.
        # B2L rows: 1, 2. Cols: 0, 1.
        # C cols: 0, 1.
        strassen_2x2_2x2_generic(
            a_row_offset, 1,
            1, 0,
            c_row_offset, c_col_offset
        )
        
        # Sub-part 2b: A2 * B2R (2x2 * 2x1)
        # A2 cols: 1, 2.
        # B2R rows: 1, 2. Col: 2.
        # C col: 2.
        schoolbook_2x2_2x1_generic(
            a_row_offset, 1,
            1, 2,
            c_row_offset, c_col_offset + 2
        )
        
    # Main construction
    # 4 blocks of 2x5 * 5x3
    for a_row in [0, 2]:
        for b_col in [0, 3]:
            c_row = a_row
            c_col = b_col
            
            # Block: A (2x5) * B (5x3)
            # Split middle dim 5 into 2 + 3
            # A_L (2x2), A_R (2x3)
            # B_T (2x3), B_B (3x3)
            
            # Part 1: A_L (2x2) * B_T (2x3)
            # Split B_T (2x3) into B_TL (2x2) and B_TR (2x1)
            # A_L * B_TL: 2x2 * 2x2 (Strassen)
            # A_L * B_TR: 2x2 * 2x1 (Schoolbook)
            
            strassen_2x2_2x2_generic(
                a_row, 0,
                0, 0,
                c_row, c_col
            )
            
            schoolbook_2x2_2x1_generic(
                a_row, 0,
                0, 2,
                c_row, c_col + 2
            )
            
            # Part 2: A_R (2x3) * B_B (3x3)
            block_2x3_3(a_row, b_col, c_row, c_col)
            
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
