import time

def construct():
    n = 2
    m = 4
    p = 8
    
    # Strassen 2x2x2 factors
    # U: 4x7 (i,j)
    # V: 4x7 (j,k)
    # W: 4x7 (k,i)
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0
        [0, 0, 0, 0, 0, 1, 0],  # i=0,j=1
        [0, 1, 0, 0, 1, 0, 1],  # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0
        [0, 0, 0, 1, 0, 0, 1],  # j=0,k=1
        [1, 0, 1, 0, 0, 1, 0],  # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1
    ]
    
    # Expand to 2x4x2 (Rank 14)
    # U: 8x14
    # V: 8x14
    # W: 4x14
    
    U_242 = [[0]*14 for _ in range(8)]
    V_242 = [[0]*14 for _ in range(8)]
    W_242 = [[0]*14 for _ in range(4)]
    
    # Block 0: j0=0
    for i in range(2):
        for j1 in range(2):
            j = 2*j1
            idx_u = i*4 + j
            for r in range(7):
                U_242[idx_u][r] = U_strassen[i*2+j1][r]
                
    for j1 in range(2):
        for k in range(2):
            j = 2*j1
            idx_v = j*2 + k
            for r in range(7):
                V_242[idx_v][r] = V_strassen[j1*2+k][r]
                
    for k in range(2):
        for i in range(2):
            idx_w = k*2 + i
            for r in range(7):
                W_242[idx_w][r] = W_strassen[k*2+i][r]
                
    # Block 1: j0=1
    for i in range(2):
        for j1 in range(2):
            j = 2*j1 + 1
            idx_u = i*4 + j
            for r in range(7):
                U_242[idx_u][7+r] = U_strassen[i*2+j1][r]
                
    for j1 in range(2):
        for k in range(2):
            j = 2*j1 + 1
            idx_v = j*2 + k
            for r in range(7):
                V_242[idx_v][7+r] = V_strassen[j1*2+k][r]
                
    for k in range(2):
        for i in range(2):
            idx_w = k*2 + i
            for r in range(7):
                W_242[idx_w][7+r] = W_strassen[k*2+i][r]
                
    # Expand to 2x4x8 (Rank 56)
    # U: 8x56
    # V: 32x56
    # W: 16x56
    
    U_final = [[0]*56 for _ in range(8)]
    V_final = [[0]*56 for _ in range(32)]
    W_final = [[0]*56 for _ in range(16)]
    
    for q in range(4):
        offset = q * 14
        # U is same for all blocks
        for i in range(8):
            for r in range(14):
                U_final[i][offset + r] = U_242[i][r]
                
        # V
        for j in range(4):
            for k in range(2):
                k_block = 2*q + k
                idx_v = j*8 + k_block
                for r in range(14):
                    V_final[idx_v][offset + r] = V_242[j*2+k][r]
                    
        # W
        for k in range(2):
            k_block = 2*q + k
            for i in range(2):
                idx_w = k_block*2 + i
                for r in range(14):
                    W_final[idx_w][offset + r] = W_242[k*2+i][r]
                    
    return {"U": U_final, "V": V_final, "W": W_final}
