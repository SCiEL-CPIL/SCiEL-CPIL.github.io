import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define Strassen coefficients for 2x2x2
    # m1 = (a+b)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a(f-g)
    # m4 = d(h-e)
    # m5 = (a+b)h
    # m6 = (c+d)g
    # m7 = (b-d)(e+h)
    
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # U entries for A
    # V entries for B
    # W entries for C
    
    # We need to map local indices to global indices.
    # Global A is 2x4.
    # Global B is 4x7.
    # Global C is 2x7.
    
    # Block 1: A[:, 0:2] and B[0:2, :]
    # Block 2: A[:, 2:4] and B[2:4, :]
    
    def add_strassen_block(a_offset, b_row_offset, b_col_offset):
        nonlocal col_idx
        # a_offset: row offset in A (0), col offset in A (0 or 2)
        # b_row_offset: row offset in B (0 or 2)
        # b_col_offset: col offset in B (0, 2, 4, 6)
        
        # Local A is 2x2 at A[a_offset:a_offset+2, a_col_offset:a_col_offset+2]
        # Local B is 2x2 at B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        # Local C is 2x2 at C[0:2, b_col_offset:b_col_offset+2]
        
        # Strassen products
        # m1: (A00+A01)(B00+B10) -> C00, C11
        # m2: (A10+A11)(B00+B10) -> C10, C11
        # m3: A00(B01-B11) -> C01, C11
        # m4: A11(B11-B00) -> C00, C01
        # m5: (A00+A01)B11 -> C00, C10
        # m6: (A10+A11)B10 -> C10, C11
        # m7: (A01-A11)(B00+B11) -> C10, C01
        
        # Note: B indices in local 2x2:
        # B00 is B[b_row_offset, b_col_offset]
        # B01 is B[b_row_offset, b_col_offset+1]
        # B10 is B[b_row_offset+1, b_col_offset]
        # B11 is B[b_row_offset+1, b_col_offset+1]
        
        # A indices:
        # A00 is A[a_offset, a_col_offset]
        # A01 is A[a_offset, a_col_offset+1]
        # A10 is A[a_offset+1, a_col_offset]
        # A11 is A[a_offset+1, a_col_offset+1]
        
        # C indices:
        # C00 is C[0, b_col_offset]
        # C01 is C[0, b_col_offset+1]
        # C10 is C[1, b_col_offset]
        # C11 is C[1, b_col_offset+1]
        
        # We add 7 terms
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_idx
            # u_coeffs: list of (idx, val) for U
            # v_coeffs: list of (idx, val) for V
            # w_coeffs: list of (idx, val) for W
            
            for idx, val in u_coeffs:
                U[idx][col_idx] += val
            for idx, val in v_coeffs:
                V[idx][col_idx] += val
            for idx, val in w_coeffs:
                W[idx][col_idx] += val
            col_idx += 1
        
        # m1
        add_term(
            [(a_offset * 2 + a_col_offset, 1), (a_offset * 2 + a_col_offset + 1, 1)], # A00+A01
            [(b_row_offset * 7 + b_col_offset, 1), ((b_row_offset+1) * 7 + b_col_offset, 1)], # B00+B10
            [(b_col_offset * 2 + 0, 1), ((b_col_offset+1) * 2 + 1, 1)] # C00, C11
        )
        
        # m2
        add_term(
            [((a_offset+1) * 2 + a_col_offset, 1), ((a_offset+1) * 2 + a_col_offset + 1, 1)], # A10+A11
            [(b_row_offset * 7 + b_col_offset, 1), ((b_row_offset+1) * 7 + b_col_offset, 1)], # B00+B10
            [((b_col_offset) * 2 + 1, 1), ((b_col_offset+1) * 2 + 1, -1)] # C10, -C11? 
            # Strassen: C10 = m2 - m5 + m7. C11 = m1 - m2 + m3.
            # So m2 contributes +1 to C10 and -1 to C11.
        )
        
        # m3
        add_term(
            [(a_offset * 2 + a_col_offset, 1)], # A00
            [(b_row_offset * 7 + b_col_offset + 1, 1), ((b_row_offset+1) * 7 + b_col_offset + 1, -1)], # B01-B11
            [((b_col_offset+1) * 2 + 0, 1), ((b_col_offset+1) * 2 + 1, 1)] # C01, C11
        )
        
        # m4
        add_term(
            [((a_offset+1) * 2 + a_col_offset + 1, 1)], # A11
            [((b_row_offset+1) * 7 + b_col_offset + 1, 1), (b_row_offset * 7 + b_col_offset, -1)], # B11-B00
            [(b_col_offset * 2 + 0, 1), ((b_col_offset+1) * 2 + 0, -1)] # C00, -C01
        )
        
        # m5
        add_term(
            [(a_offset * 2 + a_col_offset, 1), (a_offset * 2 + a_col_offset + 1, 1)], # A00+A01
            [((b_row_offset+1) * 7 + b_col_offset + 1, 1)], # B11
            [(b_col_offset * 2 + 0, -1), ((b_col_offset) * 2 + 1, 1)] # -C00, C10
        )
        
        # m6
        add_term(
            [((a_offset+1) * 2 + a_col_offset, 1), ((a_offset+1) * 2 + a_col_offset + 1, 1)], # A10+A11
            [((b_row_offset+1) * 7 + b_col_offset, 1)], # B10
            [((b_col_offset) * 2 + 1, -1), ((b_col_offset+1) * 2 + 1, 1)] # -C10, C11
        )
        
        # m7
        add_term(
            [(a_offset * 2 + a_col_offset + 1, 1), ((a_offset+1) * 2 + a_col_offset + 1, -1)], # A01-A11
            [(b_row_offset * 7 + b_col_offset, 1), ((b_row_offset+1) * 7 + b_col_offset + 1, 1)], # B00+B11
            [((b_col_offset) * 2 + 1, 1), ((b_col_offset+1) * 2 + 0, -1)] # C10, -C01
        )
        
    def add_schoolbook_block_2x1(a_col_offset, b_row_offset, b_col_offset):
        nonlocal col_idx
        # A is 2x2 at A[0:2, a_col_offset:a_col_offset+2]
        # B is 2x1 at B[b_row_offset:b_row_offset+2, b_col_offset]
        # C is 2x1 at C[0:2, b_col_offset]
        
        # 4 terms: A00*B0, A01*B1, A10*B0, A11*B1
        
        # Term 1: A00 * B0 -> C0
        U[a_offset * 2 + a_col_offset][col_idx] = 1
        V[b_row_offset * 7 + b_col_offset][col_idx] = 1
        W[b_col_offset * 2 + 0][col_idx] = 1
        col_idx += 1
        
        # Term 2: A01 * B1 -> C0
        U[a_offset * 2 + a_col_offset + 1][col_idx] = 1
        V[(b_row_offset+1) * 7 + b_col_offset][col_idx] = 1
        W[b_col_offset * 2 + 0][col_idx] = 1
        col_idx += 1
        
        # Term 3: A10 * B0 -> C1
        U[(a_offset+1) * 2 + a_col_offset][col_idx] = 1
        V[b_row_offset * 7 + b_col_offset][col_idx] = 1
        W[b_col_offset * 2 + 1][col_idx] = 1
        col_idx += 1
        
        # Term 4: A11 * B1 -> C1
        U[(a_offset+1) * 2 + a_col_offset + 1][col_idx] = 1
        V[(b_row_offset+1) * 7 + b_col_offset][col_idx] = 1
        W[b_col_offset * 2 + 1][col_idx] = 1
        col_idx += 1

    # Process Block 1 (A[:, 0:2], B[0:2, :])
    # B cols 0,1 -> Strassen
    add_strassen_block(0, 0, 0)
    # B cols 2,3 -> Strassen
    add_strassen_block(0, 0, 2)
    # B cols 4,5 -> Strassen
    add_strassen_block(0, 0, 4)
    # B col 6 -> Schoolbook 2x1
    add_schoolbook_block_2x1(0, 0, 6)
    
    # Process Block 2 (A[:, 2:4], B[2:4, :])
    # B cols 0,1 -> Strassen
    add_strassen_block(1, 2, 0) # a_offset=1 (col 2), b_row_offset=2
    # B cols 2,3 -> Strassen
    add_strassen_block(1, 2, 2)
    # B cols 4,5 -> Strassen
    add_strassen_block(1, 2, 4)
    # B col 6 -> Schoolbook 2x1
    add_schoolbook_block_2x1(1, 2, 6)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
