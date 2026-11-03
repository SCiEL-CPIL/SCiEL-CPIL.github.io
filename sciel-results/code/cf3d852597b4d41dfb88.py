import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 via Strassen on 2x2 blocks
    # 3 blocks of columns, each 2x4x2
    # Each 2x4x2 uses 14 products (2 Strassen 2x2x2)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(3):
        # B block columns: 2k, 2k+1
        # A is full 2x4
        # Split A into A_L (cols 0,1) and A_R (cols 2,3)
        # Split B_block into B_top (rows 0,1) and B_bot (rows 2,3)
        
        # Strassen for A_L * B_top
        # A_L = [[a00, a01], [a10, a11]] (indices 0,1,4,5 in A)
        # B_top = [[b00, b01], [b10, b11]] (indices 0,1, 6,7 in B)
        
        # Strassen 7 products
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C_top = [[c00, c01], [c10, c11]]
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # We will add these 7 columns to U, V, W
        # Then add 7 more for A_R * B_bot
        
        def add_strassen(a_indices, b_indices, c_indices, col_offset):
            # a_indices: [a00, a01, a10, a11] indices in U
            # b_indices: [b00, b01, b10, b11] indices in V
            # c_indices: [c00, c01, c10, c11] indices in W
            nonlocal col
            
            # M1
            U[a_indices[0]][col] += 1
            U[a_indices[3]][col] += 1
            V[b_indices[0]][col] += 1
            V[b_indices[3]][col] += 1
            W[c_indices[0]][col] += 1
            W[c_indices[3]][col] += 1
            col += 1
            
            # M2
            U[a_indices[2]][col] += 1
            U[a_indices[3]][col] += 1
            V[b_indices[0]][col] += 1
            W[c_indices[2]][col] += 1
            W[c_indices[3]][col] -= 1
            col += 1
            
            # M3
            U[a_indices[0]][col] += 1
            V[b_indices[1]][col] += 1
            V[b_indices[3]][col] -= 1
            W[c_indices[1]][col] += 1
            W[c_indices[3]][col] += 1
            col += 1
            
            # M4
            U[a_indices[3]][col] += 1
            V[b_indices[2]][col] += 1
            V[b_indices[0]][col] -= 1
            W[c_indices[0]][col] += 1
            W[c_indices[2]][col] += 1
            col += 1
            
            # M5
            U[a_indices[0]][col] += 1
            U[a_indices[1]][col] += 1
            V[b_indices[3]][col] += 1
            W[c_indices[0]][col] -= 1
            W[c_indices[1]][col] += 1
            col += 1
            
            # M6
            U[a_indices[2]][col] += 1
            U[a_indices[0]][col] -= 1
            V[b_indices[0]][col] += 1
            V[b_indices[1]][col] += 1
            W[c_indices[3]][col] += 1
            col += 1
            
            # M7
            U[a_indices[1]][col] += 1
            U[a_indices[3]][col] -= 1
            V[b_indices[2]][col] += 1
            V[b_indices[3]][col] += 1
            W[c_indices[0]][col] += 1
            col += 1
            
            return col

        # Indices for A_L
        # A is 2x4. i in 0..1, j in 0..3.
        # A_L cols 0,1.
        # a00: i=0,j=0 -> 0
        # a01: i=0,j=1 -> 1
        # a10: i=1,j=0 -> 4
        # a11: i=1,j=1 -> 5
        a_L = [0, 1, 4, 5]
        
        # Indices for B_top
        # B is 4x6. j in 0..3, k in 0..5.
        # Block k: cols 2k, 2k+1.
        # B_top rows 0,1.
        # b00: j=0, k=2k -> 0*6 + 2k
        # b01: j=0, k=2k+1 -> 0*6 + 2k+1
        # b10: j=1, k=2k -> 1*6 + 2k
        # b11: j=1, k=2k+1 -> 1*6 + 2k+1
        b_top = [2*k, 2*k+1, 6+2*k, 6+2*k+1]
        
        # Indices for C_top
        # C is 2x6. i in 0..1, k in 0..5.
        # W index: k*n + i = 2*k + i.
        # c00: i=0, k=2k -> 2*(2k) + 0 = 4k
        # c01: i=0, k=2k+1 -> 2*(2k+1) + 0 = 4k+2
        # c10: i=1, k=2k -> 2*(2k) + 1 = 4k+1
        # c11: i=1, k=2k+1 -> 2*(2k+1) + 1 = 4k+3
        c_top = [4*k, 4*k+2, 4*k+1, 4*k+3]
        
        col = add_strassen(a_L, b_top, c_top, col)
        
        # Strassen for A_R * B_bot
        # A_R cols 2,3.
        # a00: i=0,j=2 -> 2
        # a01: i=0,j=3 -> 3
        # a10: i=1,j=2 -> 6
        # a11: i=1,j=3 -> 7
        a_R = [2, 3, 6, 7]
        
        # B_bot rows 2,3.
        # b00: j=2, k=2k -> 2*6 + 2k
        # b01: j=2, k=2k+1 -> 2*6 + 2k+1
        # b10: j=3, k=2k -> 3*6 + 2k
        # b11: j=3, k=2k+1 -> 3*6 + 2k+1
        b_bot = [12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1]
        
        # C is same as c_top
        c_bot = c_top
        
        col = add_strassen(a_R, b_bot, c_bot, col)
        
    return {"U": U, "V": V, "W": W}
