import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 97: Strassen 4x4x4 (49) + Schoolbook 4x4x3 (48)
    rank_strassen = 49
    rank_school = 4 * 4 * 3
    rank = rank_strassen + rank_school
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 ---
    # We use a recursive Strassen decomposition.
    # Since we need explicit vectors, we generate them.
    
    def generate_strassen_4x4():
        # Returns U, V, W for 4x4x4 Strassen (Rank 49)
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # We define the 7 block-level Strassen terms.
        # Each term is a product of two 2x2 matrices.
        # We use Strassen for each 2x2 multiplication.
        
        # Strassen 2x2 vectors
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1
        U2[0][0]=1; U2[3][0]=1
        V2[0][0]=1; V2[3][0]=1
        W2[0][0]=1; W2[3][0]=1
        # M2
        U2[2][1]=1; U2[3][1]=1
        V2[0][1]=1
        W2[2][1]=1; W2[3][1]=-1
        # M3
        U2[0][2]=1
        V2[2][2]=1; V2[3][2]=-1
        W2[1][2]=1; W2[3][2]=1
        # M4
        U2[3][3]=1
        V2[0][3]=-1; V2[1][3]=1
        W2[0][3]=1; W2[2][3]=1
        # M5
        U2[0][4]=1; U2[1][4]=1
        V2[3][4]=1
        W2[0][4]=-1; W2[1][4]=1
        # M6
        U2[0][5]=-1; U2[2][5]=1
        V2[0][5]=1; V2[1][5]=1
        W2[3][5]=1
        # M7
        U2[1][6]=1; U2[3][6]=-1
        V2[2][6]=1; V2[3][6]=1
        W2[0][6]=1
        
        # Block definitions
        # k: (A_blocks, B_blocks, C_blocks)
        # A_blocks: dict {(br, bc): sign}
        block_defs = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        def map_idx(br, bc, lr, lc, dim):
            r = br * 2 + lr
            c = bc * 2 + lc
            return r * dim + c

        col = 0
        for k in range(7):
            a_blocks, b_blocks, c_blocks = block_defs[k]
            
            for j in range(7):
                # For this sub-term j, we add contributions to U, V, W
                # U: Sum over all A elements involved
                for br_a, bc_a in a_blocks:
                    sign_a = a_blocks[(br_a, bc_a)]
                    for lr_a in range(2):
                        for lc_a in range(2):
                            val = U2[lr_a*2+lc_a][j]
                            if val != 0:
                                idx_a = map_idx(br_a, bc_a, lr_a, lc_a, 4)
                                U4[idx_a][col] += sign_a * val
                                
                for br_b, bc_b in b_blocks:
                    sign_b = b_blocks[(br_b, bc_b)]
                    for lr_b in range(2):
                        for lc_b in range(2):
                            val = V2[lr_b*2+lc_b][j]
                            if val != 0:
                                idx_b = map_idx(br_b, bc_b, lr_b, lc_b, 4)
                                V4[idx_b][col] += sign_b * val
                                
                for br_c, bc_c in c_blocks:
                    sign_c = c_blocks[(br_c, bc_c)]
                    for lr_c in range(2):
                        for lc_c in range(2):
                            val = W2[lr_c*2+lc_c][j]
                            if val != 0:
                                idx_c = map_idx(br_c, bc_c, lr_c, lc_c, 4)
                                W4[idx_c][col] += sign_c * val
                                
                col += 1
                
        return U4, V4, W4

    U_s, V_s, W_s = generate_strassen_4x4()
    
    # Fill U, V, W for the first 49 columns
    for i in range(16):
        for k in range(49):
            U[i][k] = U_s[i][k]
            V[i][k] = V_s[i][k]
            # W for Strassen part corresponds to C (4x4)
            # W index: k*n + i -> k*4 + i
            # W_s is 16x49 (C is 4x4)
            # So W_s[i][k] goes to W[i][k]
            W[i][k] = W_s[i][k]
            
    # --- Part 2: Schoolbook 4x4x3 ---
    # Columns 4, 5, 6 of B
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # B index: j*7 + (k+4)
                b_idx = j * 7 + (k + 4)
                # C index: (k+4)*4 + i
                c_idx = (k + 4) * 4 + i
                # A index: i*4 + j
                a_idx = i * 4 + j
                
                U[a_idx][col_offset] = 1
                V[b_idx][col_offset] = 1
                W[c_idx][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
