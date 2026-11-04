import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using 6 Strassen blocks
    # A is 2x4, B is 4x6, C is 2x6
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a-d)(h-g)
    # P2 = (a+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b+d)(g-h)
    
    # Indices for 2x2 matrix flattened row-major:
    # 0: 0,0  1: 0,1
    # 2: 1,0  3: 1,1
    
    # U (4x7) for A
    U_s = [
        [0, 1, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 0, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, 1]   # d
    ]
    
    # V (4x7) for B
    V_s = [
        [-1, 1, 0, -1, 0, 1, -1], # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [1, 0, 0, 1, 0, 0, 1],    # g
        [-1, 0, -1, 0, 1, 0, -1]  # h
    ]
    
    # W (4x7) for C
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8x42
    V = [[0] * rank for _ in range(m * p)] # 24x42
    W = [[0] * rank for _ in range(p * n)] # 12x42
    
    col_offset = 0
    
    # Loop over the two row-blocks of A (and corresponding row-blocks of B)
    # A_block 0: rows 0,1 (all rows), cols 0,1
    # A_block 1: rows 0,1 (all rows), cols 2,3
    
    for a_block in range(2):
        # A indices: i in 0..1, j in a_block*2 .. a_block*2+1
        # B indices: j in a_block*2 .. a_block*2+1, k in 0..5
        
        for b_col_block in range(3):
            # B cols: k in b_col_block*2 .. b_col_block*2+1
            
            # Map Strassen U to global U
            for i in range(2):
                for j_local in range(2):
                    j_global = a_block * 2 + j_local
                    u_idx = i * m + j_global
                    for r in range(7):
                        U[u_idx][col_offset + r] = U_s[i * 2 + j_local][r]
                        
            # Map Strassen V to global V
            for j_local in range(2):
                j_global = a_block * 2 + j_local
                for k_local in range(2):
                    k_global = b_col_block * 2 + k_local
                    v_idx = j_global * p + k_global
                    for r in range(7):
                        V[v_idx][col_offset + r] = V_s[j_local * 2 + k_local][r]
                        
            # Map Strassen W to global W
            # W index is k*n + i
            for i in range(2):
                for k_local in range(2):
                    k_global = b_col_block * 2 + k_local
                    w_idx = k_global * n + i
                    for r in range(7):
                        W[w_idx][col_offset + r] = W_s[i * 2 + k_local][r] # Wait, W_s index is i*2 + k_local?
                        # W_s is 4x7. Rows 0,1 are i=0. Rows 2,3 are i=1.
                        # C00 (i=0, k=0) -> W_s[0]
                        # C01 (i=0, k=1) -> W_s[1]
                        # C10 (i=1, k=0) -> W_s[2]
                        # C11 (i=1, k=1) -> W_s[3]
                        # So W_s index is i*2 + k_local. Correct.
                        
            col_offset += 7
            
    return {"U": U, "V": V, "W": W}
