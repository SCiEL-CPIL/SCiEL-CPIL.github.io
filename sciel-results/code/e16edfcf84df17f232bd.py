import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen Decomposition
    u_222 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_222 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_222 = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    strassen_blocks = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    for k in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_blocks[k]
        for l in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            for (r, c), coeff_u in u_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = 2*lr + lc
                        global_idx = 4*r + 2*lr + lc
                        u_vec[global_idx] += coeff_u * u_222[l][local_idx]
                        
            for (r, c), coeff_v in v_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = 2*lr + lc
                        global_idx = 4*r + 2*lr + lc
                        v_vec[global_idx] += coeff_v * v_222[l][local_idx]
                        
            for (r, c), coeff_w in w_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = 2*lr + lc
                        global_idx = 4*r + 2*lr + lc
                        w_vec[global_idx] += coeff_w * w_222[l][local_idx]
                        
            U_444.append(u_vec)
            V_444.append(v_vec)
            W_444.append(w_vec)
            
    U_441 = []
    V_441 = []
    W_441 = []
    
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 4
            w_vec = [0] * 4
            
            u_vec[i*4 + j] = 1
            v_vec[j] = 1
            w_vec[i] = 1
            
            U_441.append(u_vec)
            V_441.append(v_vec)
            W_441.append(w_vec)
            
    rank_444 = len(U_444)
    rank_441 = len(U_441)
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)]
    W = [[0] * total_rank for _ in range(20)]
    
    for c in range(rank_444):
        for i in range(16):
            U[i][c] = U_444[c][i]
        for j in range(16):
            V[j][c] = V_444[c][j]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][c] = W_444[c][k*4 + i]
                
    for c in range(rank_441):
        col_idx = rank_444 + c
        for i in range(16):
            U[i][col_idx] = U_441[c][i]
        for j in range(4):
            V[j][col_idx] = V_441[c][j]
        for i in range(4):
            W[4*4 + i][col_idx] = W_441[c][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
