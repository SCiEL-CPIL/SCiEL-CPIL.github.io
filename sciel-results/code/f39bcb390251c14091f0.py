import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target rank 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2 * 2x2
    # x_idx: (r, c) in A (2x2 block)
    # y_idx: (r, c) in B (2x2 block)
    # z_idx: (r, c) in C (2x2 block)
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A block indices
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        # B block indices
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        # C block indices
        c00 = c_r0 * p + c_c0
        c01 = c_r0 * p + c_c0 + 1
        c10 = (c_r0 + 1) * p + c_c0
        c11 = (c_r0 + 1) * p + c_c0 + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (a10 + a11) b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a00 (b10 - b11)
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = a11 (b01 - b00)
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00 + a01) b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

    # Helper to add a trivial 2x2 * 2x1 block
    # x_idx: (r, c) in A (2x2 block, but only 2 cols used? No, 2x2 A * 2x1 B)
    # Actually A is 2x2, B is 2x1.
    def add_trivial_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A block indices (2x2)
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        # B block indices (2x1)
        b00 = b_r0 * p + b_c0
        b10 = (b_r0 + 1) * p + b_c0
        
        # C block indices (2x1)
        c00 = c_r0 * p + c_c0
        c10 = (c_r0 + 1) * p + c_c0
        
        # T1 = a00 b00
        U[a00][col] += 1
        V[b00][col] += 1
        W[c00][col] += 1
        col += 1
        
        # T2 = a01 b10
        U[a01][col] += 1
        V[b10][col] += 1
        W[c00][col] += 1
        col += 1
        
        # T3 = a10 b00
        U[a10][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # T4 = a11 b10
        U[a11][col] += 1
        V[b10][col] += 1
        W[c10][col] += 1
        col += 1

    # Process A_L B_L
    # A_L is rows 0-1, cols 0-1 of A
    # B_L is rows 0-1, cols 0-4 of B
    # C_L is rows 0-1, cols 0-4 of C
    
    # Split B_L into B_L1 (cols 0-1) and B_L2 (cols 2-4)
    # B_L1 is 2x2. B_L2 is 2x3.
    
    # Part 1: A_L * B_L1 (2x2 * 2x2) -> C_L1 (2x2, cols 0-1)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Part 2: A_L * B_L2 (2x2 * 2x3) -> C_L2 (2x3, cols 2-4)
    # Split B_L2 into B_L2a (cols 2-3, 2x2) and B_L2b (col 4, 2x1)
    
    # Part 2a: A_L * B_L2a (2x2 * 2x2) -> C_L2a (2x2, cols 2-3)
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # Part 2b: A_L * B_L2b (2x2 * 2x1) -> C_L2b (2x1, col 4)
    add_trivial_2x1(0, 0, 0, 4, 0, 4)
    
    # Process A_R B_R
    # A_R is rows 0-1, cols 2-3 of A
    # B_R is rows 2-3, cols 0-4 of B
    # C_R is rows 0-1, cols 0-4 of C
    
    # Split B_R into B_R1 (cols 0-1) and B_R2 (cols 2-4)
    
    # Part 1: A_R * B_R1 (2x2 * 2x2) -> C_R1 (2x2, cols 0-1)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: A_R * B_R2 (2x2 * 2x3) -> C_R2 (2x3, cols 2-4)
    # Split B_R2 into B_R2a (cols 2-3, 2x2) and B_R2b (col 4, 2x1)
    
    # Part 2a: A_R * B_R2a (2x2 * 2x2) -> C_R2a (2x2, cols 2-3)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Part 2b: A_R * B_R2b (2x2 * 2x1) -> C_R2b (2x1, col 4)
    add_trivial_2x1(0, 2, 2, 4, 0, 4)
    
    if col != rank:
        # If col != rank, we have an error in counting
        # But we expect 36
        pass

    return {"U": U, "V": V, "W": W}
