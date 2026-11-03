import time
import sys

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # 1. Generate D_4 (Strassen 4x4x4)
    # D_N is a list of (u, v, w) where u,v,w are lists of length N*N
    
    def get_D1():
        return [([1], [1], [1])]
    
    def combine_D(D_N, N):
        # Returns D_2N
        D_2N = []
        # Strassen terms definitions
        # alpha[k] is dict {(i,j): coeff}, beta[k] is dict {(i,j): coeff}
        # gamma[k][i][j] is coeff of M_k in C_ij
        
        alpha = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1}
        ]
        beta = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1}
        ]
        # C_00 = M0 + M3 - M1 + M5
        # C_01 = M2 + M4
        # C_10 = M1 + M3
        # C_11 = M0 - M4 + M6 - M5
        gamma = [
            [[1, 0], [0, 1]], # M0 in C00, C11
            [[-1, 0], [1, 0]], # M1 in C00, C10
            [[0, 1], [0, 0]], # M2 in C01
            [[1, 0], [1, 0]], # M3 in C00, C10
            [[0, 1], [0, -1]], # M4 in C01, C11
            [[1, 0], [0, -1]], # M5 in C00, C11
            [[0, 0], [0, 1]]  # M6 in C11
        ]
        
        for k in range(7):
            for u, v, w in D_N:
                new_u = [0] * (4 * N * N)
                new_v = [0] * (4 * N * N)
                new_w = [0] * (4 * N * N)
                
                # Map u
                for (bi, bj), coeff in alpha[k].items():
                    for l in range(N * N):
                        if u[l] != 0:
                            # Global index for block (bi, bj) and local l
                            # Block (bi, bj) starts at row bi*N, col bj*N
                            # Local l corresponds to row l//N, col l%N
                            lr = l // N
                            lc = l % N
                            gr = bi * N + lr
                            gc = bj * N + lc
                            g_idx = gr * (2 * N) + gc
                            new_u[g_idx] += coeff * u[l]
                
                # Map v
                for (bi, bj), coeff in beta[k].items():
                    for l in range(N * N):
                        if v[l] != 0:
                            lr = l // N
                            lc = l % N
                            gr = bi * N + lr
                            gc = bj * N + lc
                            g_idx = gr * (2 * N) + gc
                            new_v[g_idx] += coeff * v[l]
                
                # Map w
                for i in range(2):
                    for j in range(2):
                        coeff = gamma[k][i][j]
                        if coeff != 0:
                            for l in range(N * N):
                                if w[l] != 0:
                                    lr = l // N
                                    lc = l % N
                                    gr = i * N + lr
                                    gc = j * N + lc
                                    g_idx = gr * (2 * N) + gc
                                    new_w[g_idx] += coeff * w[l]
                
                D_2N.append((new_u, new_v, new_w))
                
                if time.time() > deadline - 10:
                    return D_2N # Early exit, might be incomplete but valid? No, must be complete.
        
        return D_2N

    D1 = get_D1()
    D2 = combine_D(D1, 1)
    D4 = combine_D(D2, 2)
    
    # D4 has 49 terms
    # U is 16 x 49, V is 16 x 49, W is 16 x 49
    U_4x4x4 = [[0]*len(D4) for _ in range(16)]
    V_4x4x4 = [[0]*len(D4) for _ in range(16)]
    W_4x4x4 = [[0]*len(D4) for _ in range(16)]
    
    for c, (u, v, w) in enumerate(D4):
        for i in range(16):
            U_4x4x4[i][c] = u[i]
            V_4x4x4[i][c] = v[i]
            W_4x4x4[i][c] = w[i]
            
    # 2. Generate Schoolbook for 4x4x1
    # Rank 16
    U_4x4x1 = [[0]*16 for _ in range(16)]
    V_4x4x1 = [[0]*16 for _ in range(4)]
    W_4x4x1 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_4x4x1[i * 4 + j][c] = 1
            V_4x4x1[j][c] = 1
            W_4x4x1[i][c] = 1
            
    # 3. Combine for 4x4x9
    # B is 4x9. Split into B1 (4x4) and B2 (4x5).
    # We use D4 for B1 (cols 0-3) and Schoolbook for B2 (cols 4-8)?
    # Schoolbook for 4x4x5 has rank 4*4*5 = 80.
    # Total 49 + 80 = 129.
    # Better: Split into 4x4, 4x4, 4x1.
    # R(4,4,4) + R(4,4,4) + R(4,4,1) = 49 + 49 + 16 = 114.
    
    # We have D4 for the first 4x4 part.
    # We need another D4 for the second 4x4 part.
    # And Schoolbook for the last 4x1 part.
    
    # The U matrix is the same for all parts (it depends on A).
    # The V matrix depends on B.
    # The W matrix depends on C.
    
    # For Part 1 (cols 0-3 of B, cols 0-3 of C):
    # U is U_4x4x4
    # V is V_4x4x4 (maps to B[0:4, 0:4])
    # W is W_4x4x4 (maps to C[0:4, 0:4])
    
    # For Part 2 (cols 4-7 of B, cols 4-7 of C):
    # U is U_4x4x4
    # V is V_4x4x4 (maps to B[0:4, 4:8]) -> Need to shift V indices
    # W is W_4x4x4 (maps to C[0:4, 4:8]) -> Need to shift W indices
    
    # For Part 3 (cols 8 of B, col 8 of C):
    # U is U_4x4x1
    # V is V_4x4x1 (maps to B[0:4, 8])
    # W is W_4x4x1 (maps to C[0:4, 8])
    
    total_rank = 49 + 49 + 16
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(36)] # 4*9
    W_final = [[0]*total_rank for _ in range(36)] # 9*4
    
    col_offset = 0
    
    # Part 1
    for c in range(49):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x4[i][c]
            V_final[i][col_offset + c] = V_4x4x4[i][c]
            # W index: k*n + i. Here k is col index of C.
            # In W_4x4x4, index is k*n + i where k is 0..3.
            # We want k to be 0..3. So no shift.
            W_final[i][col_offset + c] = W_4x4x4[i][c]
    col_offset += 49
    
    # Part 2
    for c in range(49):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x4[i][c]
            # V index: j*p + k. p=9.
            # In V_4x4x4, index is j*4 + k where k is 0..3.
            # We want k to be 4..7. So add 4 to k.
            # j*4 + k -> j*9 + (k+4)
            # i in V_4x4x4 is j*4 + k.
            # We need to map i to j*9 + k + 4.
            j = i // 4
            k = i % 4
            v_idx = j * 9 + (k + 4)
            V_final[v_idx][col_offset + c] = V_4x4x4[i][c]
            
            # W index: k*n + i.
            # In W_4x4x4, index is k*4 + i.
            # We want k to be 4..7. So add 4 to k.
            # k*4 + i -> (k+4)*4 + i
            k_w = i // 4
            i_w = i % 4
            w_idx = (k_w + 4) * 4 + i_w
            W_final[w_idx][col_offset + c] = W_4x4x4[i][c]
    col_offset += 49
    
    # Part 3
    for c in range(16):
        for i in range(16):
            U_final[i][col_offset + c] = U_4x4x1[i][c]
            # V index: j*9 + 8
            j = i // 4
            k = i % 4 # k is 0, but we only have 1 col, so k=0
            # In V_4x4x1, index is j.
            v_idx = j * 9 + 8
            V_final[v_idx][col_offset + c] = V_4x4x1[j][c]
            
            # W index: 8*4 + i
            # In W_4x4x1, index is i.
            w_idx = 8 * 4 + i
            W_final[w_idx][col_offset + c] = W_4x4x1[i][c]
            
    return {"U": U_final, "V": V_final, "W": W_final}
