import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimation
    # 4 Strassen blocks (7 each) = 28
    # 2 Direct 2x2x1 blocks (4 each) = 8
    # 1 Direct 2x1x10 block (10) = 10
    # Total = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 tensor term
    # u: list of (index, coeff) for U
    # v: list of (index, coeff) for V
    # w: list of (index, coeff) for W
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff
        col += 1
        
    # Helper for Strassen 2x2x2
    # a_row, a_col_offset for A (2x2)
    # b_row_offset, b_col_offset for B (2x2)
    # c_row_offset, c_col_offset for C (2x2)
    # A is n x m, B is m x p, C is n x p
    # Indices:
    # A[i, j] -> U[i*m + j]
    # B[j, k] -> V[j*p + k]
    # C[i, k] -> W[k*n + i]
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A entries:
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        # B entries:
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        # C entries:
        c00 = c_c0 * n + c_r0
        c01 = (c_c0 + 1) * n + c_r0
        c10 = c_c0 * n + c_r0 + 1
        c11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c01, 1), (c11, 1)])
        # Wait, W mapping: W[k*n+i, r].
        # C[i, k] corresponds to W[k*n + i].
        # M1 contributes to C[0,0], C[0,1]? No.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        # C[0,0] = M1... -> W[0*n + 0] = W[0]
        # C[0,1] = M3+M5 -> W[1*n + 0] = W[2] (since n=2)
        # C[1,0] = M2+M4 -> W[0*n + 1] = W[1]
        # C[1,1] = M1... -> W[1*n + 1] = W[3]
        
        # Let's re-verify W index mapping.
        # Task: W[k*n+i, r].
        # C[i, k] is the entry.
        # So for C[0,0], k=0, i=0 -> W[0].
        # For C[0,1], k=1, i=0 -> W[1*2 + 0] = W[2].
        # For C[1,0], k=0, i=1 -> W[0*2 + 1] = W[1].
        # For C[1,1], k=1, i=1 -> W[1*2 + 1] = W[3].
        
        # My add_term above used c00, c01 etc as if they were linear indices.
        # I need to pass the W indices correctly.
        
        # Let's refine add_strassen to use correct W indices.
        pass

    # Corrected Strassen
    def add_strassen_correct(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        # W indices: W[k*n + i]
        # c_r0 is i_offset, c_c0 is k_offset
        w00 = c_c0 * n + c_r0       # C[0,0]
        w01 = (c_c0 + 1) * n + c_r0 # C[0,1]
        w10 = c_c0 * n + c_r0 + 1   # C[1,0]
        w11 = (c_c0 + 1) * n + c_r0 + 1 # C[1,1]
        
        # M1 = (a00 + a11)(b00 + b11)
        # Contributes to C[0,0] (+1), C[1,1] (+1)
        add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(w00, 1), (w11, 1)])
        
        # M2 = (a10 + a11)b00
        # Contributes to C[1,0] (+1), C[1,1] (-1)
        add_term([(a10, 1), (a11, 1)], [(b00, 1)], [(w10, 1), (w11, -1)])
        
        # M3 = a00(b01 - b11)
        # Contributes to C[0,1] (+1), C[1,1] (+1)
        add_term([(a00, 1)], [(b01, 1), (b11, -1)], [(w01, 1), (w11, 1)])
        
        # M4 = a11(b10 - b00)
        # Contributes to C[0,0] (+1), C[1,0] (+1)
        add_term([(a11, 1)], [(b10, 1), (b00, -1)], [(w00, 1), (w10, 1)])
        
        # M5 = (a00 + a01)b11
        # Contributes to C[0,0] (-1), C[0,1] (+1)
        add_term([(a00, 1), (a01, 1)], [(b11, 1)], [(w00, -1), (w01, 1)])
        
        # M6 = (a10 - a00)(b00 + b01)
        # Contributes to C[1,1] (+1)
        add_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(w11, 1)])
        
        # M7 = (a01 - a11)(b10 + b11)
        # Contributes to C[0,0] (+1)
        add_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(w00, 1)])

    # Helper for Direct 2x2x1
    def add_direct_2x2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        b00 = b_r0 * p + b_c0
        b10 = (b_r0 + 1) * p + b_c0
        
        w00 = c_c0 * n + c_r0       # C[0,0]
        w10 = c_c0 * n + c_r0 + 1   # C[1,0]
        
        # C[0,0] = a00*b00 + a01*b10
        add_term([(a00, 1)], [(b00, 1)], [(w00, 1)])
        add_term([(a01, 1)], [(b10, 1)], [(w00, 1)])
        
        # C[1,0] = a10*b00 + a11*b10
        add_term([(a10, 1)], [(b00, 1)], [(w10, 1)])
        add_term([(a11, 1)], [(b10, 1)], [(w10, 1)])

    # Helper for Direct 2x1x10
    def add_direct_2x1x10(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A is 2x1, B is 1x10, C is 2x10
        a0 = a_r0 * m + a_c0
        a1 = (a_r0 + 1) * m + a_c0
        
        # B is 1x10 starting at b_r0, b_c0
        # C is 2x10 starting at c_r0, c_c0
        
        for k in range(10):
            b_idx = b_r0 * p + (b_c0 + k)
            w0_idx = (c_c0 + k) * n + c_r0       # C[0, k]
            w1_idx = (c_c0 + k) * n + c_r0 + 1   # C[1, k]
            
            # C[0, k] = a0 * b[k]
            add_term([(a0, 1)], [(b_idx, 1)], [(w0_idx, 1)])
            
            # C[1, k] = a1 * b[k]
            add_term([(a1, 1)], [(b_idx, 1)], [(w1_idx, 1)])

    # Execute blocks
    
    # Block 1: A2 (rows 0-1, cols 0-1 of A) * B2 (rows 0-1 of B)
    # A2 is 2x2. B2 is 2x10.
    # Split B2 into B2L (cols 0-4) and B2R (cols 5-9)
    
    # B2L (cols 0-4)
    # Split into 2x2, 2x2, 2x1
    # B2L1: cols 0-1
    add_strassen_correct(0, 0, 0, 0, 0, 0)
    # B2L2: cols 2-3
    add_strassen_correct(0, 0, 0, 2, 0, 2)
    # B2L3: col 4
    add_direct_2x2x1(0, 0, 0, 4, 0, 4)
    
    # B2R (cols 5-9)
    # Split into 2x2, 2x2, 2x1
    # B2R1: cols 5-6
    add_strassen_correct(0, 0, 0, 5, 0, 5)
    # B2R2: cols 7-8
    add_strassen_correct(0, 0, 0, 7, 0, 7)
    # B2R3: col 9
    add_direct_2x2x1(0, 0, 0, 9, 0, 9)
    
    # Block 2: A1 (rows 0-1, col 2 of A) * B1 (row 2 of B)
    # A1 is 2x1. B1 is 1x10.
    add_direct_2x1x10(0, 2, 2, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
