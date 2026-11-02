import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_idx: list of 4 indices in U for A[0,0], A[0,1], A[1,0], A[1,1]
    # b_idx: list of 4 indices in V for B[0,0], B[0,1], B[1,0], B[1,1]
    # c_idx: list of 4 indices in W for C[0,0], C[0,1], C[1,0], C[1,1]
    def add_strassen_block(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (A00+A11)(B00+B11) -> C00+, C11+
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10+
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M3 = A00(B10-B11) -> C01+
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M4 = A11(B01-B00) -> C00-, C11+
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] -= 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M5 = (A00+A10)B11 -> C00-, C11+
        U[a_idx[0]][col] += 1
        U[a_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M6 = (A11+A10)(B00-B10) -> C10+
        U[a_idx[3]][col] += 1
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] -= 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M7 = (A00+A01)(B11-B10) -> C00-, C01+
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        V[b_idx[2]][col] -= 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1

    # Part 1: A1 (rows 0,1) * B
    # A1 is 2x4. Split into A1L (cols 0,1) and A1R (cols 2,3).
    # B is 4x8. Split into BL (rows 0,1) and BR (rows 2,3).
    
    # A1L indices in U:
    # A[0,0] -> 0*4+0=0
    # A[0,1] -> 0*4+1=1
    # A[1,0] -> 1*4+0=4
    # A[1,1] -> 1*4+1=5
    a1l_idx = [0, 1, 4, 5]
    
    # A1R indices in U:
    # A[0,2] -> 0*4+2=2
    # A[0,3] -> 0*4+3=3
    # A[1,2] -> 1*4+2=6
    # A[1,3] -> 1*4+3=7
    a1r_idx = [2, 3, 6, 7]
    
    # B indices in V:
    # B[j,k] -> j*8+k
    # BL: rows 0,1. Cols 0-1, 2-3, 4-5, 6-7
    # BR: rows 2,3. Cols 0-1, 2-3, 4-5, 6-7
    
    for block_col in range(4):
        # Cols for this block: 2*block_col, 2*block_col+1
        c_start = 2 * block_col
        
        # B_L block indices
        b_l_idx = [
            0 * 8 + c_start,     # B[0, c_start]
            0 * 8 + c_start + 1, # B[0, c_start+1]
            1 * 8 + c_start,     # B[1, c_start]
            1 * 8 + c_start + 1  # B[1, c_start+1]
        ]
        
        # C indices for A1L * BL_block
        # C[i, k] -> k*3+i
        # i=0,1. k=c_start, c_start+1
        c_idx = [
            c_start * 3 + 0,     # C[0, c_start]
            c_start * 3 + 1,     # C[0, c_start+1]
            c_start * 3 + 2,     # C[1, c_start]
            c_start * 3 + 3      # C[1, c_start+1]
        ]
        
        add_strassen_block(a1l_idx, b_l_idx, c_idx)
        
        # B_R block indices
        b_r_idx = [
            2 * 8 + c_start,     # B[2, c_start]
            2 * 8 + c_start + 1, # B[2, c_start+1]
            3 * 8 + c_start,     # B[3, c_start]
            3 * 8 + c_start + 1  # B[3, c_start+1]
        ]
        
        # C indices for A1R * BR_block are the same C positions
        add_strassen_block(a1r_idx, b_r_idx, c_idx)
        
    # Part 2: A2 (row 2) * B
    # A2 is 1x4.
    # U indices: A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # V indices: B[j,k]
    # W indices: C[2,k] -> k*3+2
    
    for j in range(4):
        for k in range(8):
            u_idx = 2 * 4 + j
            v_idx = j * 8 + k
            w_idx = k * 3 + 2
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
