import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 2x2x2 Strassen factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 4x4x4 Strassen factors
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    block_defs = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for k in range(7):
        x_blocks, y_blocks, c_blocks = block_defs[k]
        for s in range(7):
            rank_idx = k * 7 + s
            
            # U
            for i2 in range(4):
                u_val = U2[i2][s]
                if u_val == 0: continue
                i2_row = i2 // 2
                i2_col = i2 % 2
                
                for br, bc, scale in x_blocks:
                    if br == i2_row and bc == i2_col:
                        val = u_val * scale
                        if val == 0: continue
                        for r in range(2):
                            for c in range(2):
                                global_i = br * 2 + r
                                global_j = bc * 2 + c
                                U4[global_i * 4 + global_j][rank_idx] += val
            
            # V
            for j2 in range(4):
                v_val = V2[j2][s]
                if v_val == 0: continue
                j2_row = j2 // 2
                j2_col = j2 % 2
                
                for br, bc, scale in y_blocks:
                    if br == j2_row and bc == j2_col:
                        val = v_val * scale
                        if val == 0: continue
                        for r in range(2):
                            for c in range(2):
                                global_i = br * 2 + r
                                global_j = bc * 2 + c
                                V4[global_i * 4 + global_j][rank_idx] += val
            
            # W
            for c2 in range(4):
                w_val = W2[c2][s]
                if w_val == 0: continue
                c2_row = c2 // 2
                c2_col = c2 % 2
                
                for br, bc, scale in c_blocks:
                    if br == c2_row and bc == c2_col:
                        val = w_val * scale
                        if val == 0: continue
                        for r in range(2):
                            for c in range(2):
                                global_i = br * 2 + r
                                global_j = bc * 2 + c
                                W4[global_j * 4 + global_i][rank_idx] += val

    # 4x4x1 Schoolbook
    U1 = [[0]*16 for _ in range(16)]
    V1 = [[0]*16 for _ in range(4)]
    W1 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U1[i * 4 + j][r] = 1
            V1[j][r] = 1
            W1[i][r] = 1

    # Combine
    # 4x4x4 (cols 0-3)
    # 4x4x4 (cols 4-7)
    # 4x4x1 (col 8)
    
    total_rank = 49 + 49 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Part 1: 4x4x4
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            V[i][r] = V4[i][r]
            W[i][r] = W4[i][r]
            
    # Part 2: 4x4x4
    for i in range(16):
        for r in range(49):
            U[i][49+r] = U4[i][r]
            # V index: j*4 + k. k is 4..7.
            # V4 index is j*4 + k_local.
            # So V[i][49+r] = V4[i][r] is correct if we shift k.
            # Wait, V4 is for 4x4x4.
            # V4[j*4+k, r].
            # We want V[j*4 + (k+4), 49+r] = V4[j*4+k, r].
            # So V[i][49+r] = V4[i][r] works if i is interpreted correctly.
            # But V4 is indexed by j*4+k.
            # So V[i][49+r] = V4[i][r] is correct.
            W[i][49+r] = W4[i][r]
            
    # Part 3: 4x4x1
    for i in range(16):
        for r in range(16):
            U[i][98+r] = U1[i][r]
            # V1 is 4x16. Index j.
            # We want V[j*4+8, 98+r] = V1[j, r].
            # So V[i][98+r] is not directly V1.
            # V has shape 36xRank.
            # V1 has shape 4x16.
            # For each j in 0..3, V[j*4+8, 98+r] = V1[j, r].
            pass
            
    # Correct V and W for Part 3
    for j in range(4):
        for r in range(16):
            V[j*4+8][98+r] = V1[j][r]
            
    for i in range(4):
        for r in range(16):
            W[i][98+r] = W1[i][r]
            
    return {"U": U, "V": V, "W": W}
