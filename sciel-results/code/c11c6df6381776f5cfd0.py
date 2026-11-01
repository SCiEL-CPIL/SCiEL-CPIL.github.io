import time
import sys
import numpy as np

def construct():
    """
    Find a rank-minimal tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses block decomposition: 3 blocks of 2x3 by 3x2.
    Each 2x3x2 block uses rank 11 decomposition (Strassen for 2x2x2 + Schoolbook for 2x1x2).
    Total rank 33.
    """
    n, m, p = 2, 3, 6
    # Split p=6 into 3 blocks of 2
    num_blocks = 3
    block_p = 2
    
    # Total rank will be num_blocks * 11 = 33
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Indices for U: i in 0..1, j in 0..2 -> idx = i*m + j
    # Indices for V: j in 0..2, k in 0..5 -> idx = j*p + k
    # Indices for W: k in 0..5, i in 0..1 -> idx = k*n + i
    
    col_offset = 0
    for block_idx in range(num_blocks):
        # k indices for this block: block_idx*2, block_idx*2 + 1
        k_start = block_idx * block_p
        
        # We decompose A (2x3) x B_block (3x2)
        # A is shared across blocks, but we assign U entries for each block's columns
        # B_block is V entries for this block's columns
        # W is for this block's output columns
        
        # Sub-decomposition 1: Strassen for A1 (2x2) x B1 (2x2)
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1] (cols 0,1 of A)
        # B1 uses B[0, k_start], B[0, k_start+1], B[1, k_start], B[1, k_start+1] (rows 0,1 of B_block)
        
        # Strassen 2x2x2 rank 7
        # M1 = (A00 + A11)(B01 + B10)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B00 - B01)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M5 + M4 - M3 - M7
        # C01 = M1 + M7
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M5
        
        # We need to map these to U, V, W entries.
        # U entries correspond to A elements.
        # V entries correspond to B elements.
        # W entries correspond to C elements.
        
        # Let's define the 7 terms for Strassen part
        # Term 1: (A00 + A11) * (B01 + B10)
        # U: A00 (0,0) = 1, A11 (1,1) = 1
        # V: B01 (0, k_start+1) = 1, B10 (1, k_start) = 1
        # W: C01 (k_start+1, 0) = 1, C11 (k_start+1, 1) = 1 ? 
        # Wait, W index is k*n + i.
        # C01 is row 0, col 1 of C_block. 
        # C_block is 2x2. 
        # Global C is 2x6.
        # Col 0 of C_block is global col k_start.
        # Col 1 of C_block is global col k_start+1.
        # Row 0 is global row 0.
        # Row 1 is global row 1.
        
        # Let's write a helper to add a rank-1 term
        def add_term(U_vals, V_vals, W_vals, col):
            for idx, val in U_vals:
                if val != 0:
                    U[idx][col] += val
            for idx, val in V_vals:
                if val != 0:
                    V[idx][col] += val
            for idx, val in W_vals:
                if val != 0:
                    W[idx][col] += val

        # Strassen Terms
        # 1
        add_term(
            [(0*m + 0, 1), (1*m + 1, 1)], # A00, A11
            [(0*p + k_start + 1, 1), (1*p + k_start, 1)], # B01, B10
            [((k_start + 1)*n + 0, 1), ((k_start + 1)*n + 1, 1)], # C01, C11
            col_offset
        )
        col_offset += 1
        
        # 2
        add_term(
            [(1*m + 0, 1), (1*m + 1, 1)], # A10, A11
            [(0*p + k_start, 1)], # B00
            [((k_start)*n + 1, 1), ((k_start + 1)*n + 1, 1)], # C10, C11
            col_offset
        )
        col_offset += 1
        
        # 3
        add_term(
            [(0*m + 0, 1)], # A00
            [(1*p + k_start, 1), (1*p + k_start + 1, -1)], # B10, B11
            [((k_start)*n + 0, -1), ((k_start + 1)*n + 0, 1)], # C00 (-M3), C10 (M3)
            col_offset
        )
        col_offset += 1
        
        # 4
        add_term(
            [(1*m + 1, 1)], # A11
            [(0*p + k_start, 1), (0*p + k_start + 1, -1)], # B00, B01
            [((k_start)*n + 0, 1), ((k_start + 1)*n + 1, -1)], # C00 (M4), C11 (-M4)
            col_offset
        )
        col_offset += 1
        
        # 5
        add_term(
            [(0*m + 0, 1), (0*m + 1, 1)], # A00, A01
            [(1*p + k_start + 1, 1)], # B11
            [((k_start)*n + 0, 1), ((k_start + 1)*n + 1, 1)], # C00 (M5), C11 (M5)
            col_offset
        )
        col_offset += 1
        
        # 6
        add_term(
            [(1*m + 0, 1), (0*m + 0, -1)], # A10, A00
            [(0*p + k_start, 1), (0*p + k_start + 1, 1)], # B00, B01
            [((k_start)*n + 1, 1), ((k_start + 1)*n + 1, 1)], # C10 (M6), C11 (M6)
            col_offset
        )
        col_offset += 1
        
        # 7
        add_term(
            [(0*m + 1, 1), (1*m + 1, -1)], # A01, A11
            [(1*p + k_start, 1), (1*p + k_start + 1, 1)], # B10, B11
            [((k_start + 1)*n + 0, -1), ((k_start + 1)*n + 1, -1)], # C01 (-M7), C11 (-M7)
            col_offset
        )
        col_offset += 1
        
        # Sub-decomposition 2: Schoolbook for a2 (2x1) x b2 (1x2)
        # a2 uses A[0,2], A[1,2] (col 2 of A)
        # b2 uses B[2, k_start], B[2, k_start+1] (row 2 of B_block)
        
        # Term 8: A02 * B20
        add_term(
            [(0*m + 2, 1)], # A02
            [(2*p + k_start, 1)], # B20
            [((k_start)*n + 0, 1)], # C00
            col_offset
        )
        col_offset += 1
        
        # Term 9: A02 * B21
        add_term(
            [(0*m + 2, 1)], # A02
            [(2*p + k_start + 1, 1)], # B21
            [((k_start + 1)*n + 0, 1)], # C01
            col_offset
        )
        col_offset += 1
        
        # Term 10: A12 * B20
        add_term(
            [(1*m + 2, 1)], # A12
            [(2*p + k_start, 1)], # B20
            [((k_start)*n + 1, 1)], # C10
            col_offset
        )
        col_offset += 1
        
        # Term 11: A12 * B21
        add_term(
            [(1*m + 2, 1)], # A12
            [(2*p + k_start + 1, 1)], # B21
            [((k_start + 1)*n + 1, 1)], # C11
            col_offset
        )
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
