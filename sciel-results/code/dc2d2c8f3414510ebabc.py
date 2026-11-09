# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin

    n, m, p = DIMENSIONS  # n=4, m=5, p=6

    # Schoolbook rank is n*m*p = 120
    rank_schoolbook = n * m * p

    # Initialize best with schoolbook
    U_best = [[0] * rank_schoolbook for _ in range(n * m)]
    V_best = [[0] * rank_schoolbook for _ in range(m * p)]
    W_best = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
    best_rank = rank_schoolbook

    # --- Local Decompositions ---
    
    # 1. Strassen 2x2x2 (Rank 7)
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1]
    ]

    # 2. Schoolbook 2x2x1 (Rank 4)
    # Indices: i in {0,1}, j in {0,1}, k in {0}
    # U len 4, V len 2, W len 2 (k*2+i)
    sb_221_u = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    sb_221_v = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    sb_221_w = [
        [1, 0], # k=0, i=0
        [1, 0], # k=0, i=0? No.
        # Term 0: i=0, j=0, k=0 -> W[0]=1
        # Term 1: i=0, j=1, k=0 -> W[0]=1
        # Term 2: i=1, j=0, k=0 -> W[1]=1
        # Term 3: i=1, j=1, k=0 -> W[1]=1
        [1, 0],
        [0, 1]
    ]
    # Wait, W index is k*n_loc + i. n_loc=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # So:
    sb_221_w = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]

    # 3. Schoolbook 2x1x3 (Rank 6)
    # Indices: i in {0,1}, j in {0}, k in {0,1,2}
    # U len 4, V len 3, W len 6 (k*2+i)
    sb_213_u = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ]
    sb_213_v = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    sb_213_w = [
        [1, 0, 0, 0, 0, 0], # k=0, i=0
        [0, 0, 1, 0, 0, 0], # k=1, i=0
        [0, 0, 0, 0, 1, 0], # k=2, i=0
        [0, 1, 0, 0, 0, 0], # k=0, i=1
        [0, 0, 0, 1, 0, 0], # k=1, i=1
        [0, 0, 0, 0, 0, 1]  # k=2, i=1
    ]

    # --- Construct Local Blocks ---

    # Block A: 2x2x3 (Rank 11)
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    # U len 4, V len 6, W len 6
    block_223_u = []
    block_223_v = []
    block_223_w = []
    
    # Add Strassen 2x2x2 (k=0,1)
    for r in range(7):
        u = strassen_u[r]
        v = [strassen_v[r][0], strassen_v[r][1], 0] # j=0,1; k=0,1,2 -> v[0..1] mapped to v[0..1], v[2..3] mapped to v[2..3]?
        # Strassen V index: j*2+k.
        # Local V index: j*3+k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1. j=1, k=0 -> 3. j=1, k=1 -> 4.
        # So v_local[0] = sv[0], v_local[1] = sv[1], v_local[3] = sv[2], v_local[4] = sv[3].
        # v_local[2] = 0.
        v_loc = [0]*6
        v_loc[0] = strassen_v[r][0]
        v_loc[1] = strassen_v[r][1]
        v_loc[3] = strassen_v[r][2]
        v_loc[4] = strassen_v[r][3]
        
        w = [0]*6
        # Strassen W index: k*2+i.
        # Local W index: k*2+i.
        # k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
        w[0] = strassen_w[r][0]
        w[1] = strassen_w[r][1]
        w[2] = strassen_w[r][2]
        w[3] = strassen_w[r][3]
        
        block_223_u.append(u)
        block_223_v.append(v_loc)
        block_223_w.append(w)
        
    # Add Schoolbook 2x2x1 (k=2)
    for r in range(4):
        u = sb_221_u[r]
        v = [0]*6
        # sb_221_v index: j.
        # Local V index: j*3+2.
        # j=0 -> 2. j=1 -> 5.
        if sb_221_v[r][0] == 1:
            v[2] = 1
        if sb_221_v[r][1] == 1:
            v[5] = 1
            
        w = [0]*6
        # sb_221_w index: k*2+i = 0*2+i = i.
        # Local W index: k*2+i = 2*2+i = 4+i.
        # i=0 -> 4. i=1 -> 5.
        if sb_221_w[r][0] == 1:
            w[4] = 1
        if sb_221_w[r][1] == 1:
            w[5] = 1
            
        block_223_u.append(u)
        block_223_v.append(v)
        block_223_w.append(w)
        
    # Block B: 2x3x3 (Rank 17)
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # U len 4, V len 9, W len 6
    block_233_u = []
    block_233_v = []
    block_233_w = []
    
    # Add Strassen 2x2x2 (j=0,1; k=0,1)
    for r in range(7):
        u = strassen_u[r]
        v = [0]*9
        # Strassen V: j*2+k.
        # Local V: j*3+k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 3. j=1, k=1 -> 4.
        v[0] = strassen_v[r][0]
        v[1] = strassen_v[r][1]
        v[3] = strassen_v[r][2]
        v[4] = strassen_v[r][3]
        
        w = [0]*6
        w[0] = strassen_w[r][0]
        w[1] = strassen_w[r][1]
        w[2] = strassen_w[r][2]
        w[3] = strassen_w[r][3]
        
        block_233_u.append(u)
        block_233_v.append(v)
        block_233_w.append(w)
        
    # Add Schoolbook 2x2x1 (j=0,1; k=2)
    for r in range(4):
        u = sb_221_u[r]
        v = [0]*9
        # sb_221_v: j.
        # Local V: j*3+2.
        # j=0 -> 2. j=1 -> 5.
        if sb_221_v[r][0] == 1:
            v[2] = 1
        if sb_221_v[r][1] == 1:
            v[5] = 1
            
        w = [0]*6
        # sb_221_w: i.
        # Local W: 2*2+i = 4+i.
        if sb_221_w[r][0] == 1:
            w[4] = 1
        if sb_221_w[r][1] == 1:
            w[5] = 1
            
        block_233_u.append(u)
        block_233_v.append(v)
        block_233_w.append(w)
        
    # Add Schoolbook 2x1x3 (j=2; k=0,1,2)
    for r in range(6):
        u = sb_213_u[r]
        v = [0]*9
        # sb_213_v: k.
        # Local V: 2*3+k = 6+k.
        if sb_213_v[r][0] == 1:
            v[6] = 1
        if sb_213_v[r][1] == 1:
            v[7] = 1
        if sb_213_v[r][2] == 1:
            v[8] = 1
            
        w = sb_213_w[r]
        
        block_233_u.append(u)
        block_233_v.append(v)
        block_233_w.append(w)

    # --- Global Assembly ---
    
    # n=4, m=5, p=6
    # n blocks: 2, 2
    # m blocks: 3, 2
    # p blocks: 3, 3
    
    # Blocks:
    # (0,0): n=2, m=3, p=3 -> Block B (Rank 17)
    # (0,1): n=2, m=2, p=3 -> Block A (Rank 11)
    # (1,0): n=2, m=3, p=3 -> Block B (Rank 17)
    # (1,1): n=2, m=2, p=3 -> Block A (Rank 11)
    
    total_rank = 17 + 11 + 17 + 11
    
    if total_rank >= best_rank:
        return {"U": U_best, "V": V_best, "W": W_best}
        
    U_new = np.zeros((n * m, total_rank), dtype=int)
    V_new = np.zeros((m * p, total_rank), dtype=int)
    W_new = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Block (0,0): i in {0,1}, j in {0,1,2}, k in {0,1,2}
    for r in range(17):
        u = block_233_u[r]
        v = block_233_v[r]
        w = block_233_w[r]
        
        for i_l in range(2):
            for j_l in range(3):
                i_g = i_l
                j_g = j_l
                U_new[i_g * m + j_g, col] += u[i_l * 2 + j_l]
                
        for j_l in range(3):
            for k_l in range(3):
                j_g = j_l
                k_g = k_l
                V_new[j_g * p + k_g, col] += v[j_l * 3 + k_l]
                
        for k_l in range(3):
            for i_l in range(2):
                k_g = k_l
                i_g = i_l
                W_new[k_g * n + i_g, col] += w[k_l * 2 + i_l]
                
        col += 1
        
    # Block (0,1): i in {0,1}, j in {3,4}, k in {0,1,2}
    # Local j in {0,1} -> Global j in {3,4}
    for r in range(11):
        u = block_223_u[r]
        v = block_223_v[r]
        w = block_223_w[r]
        
        for i_l in range(2):
            for j_l in range(2):
                i_g = i_l
                j_g = 3 + j_l
                U_new[i_g * m + j_g, col] += u[i_l * 2 + j_l]
                
        for j_l in range(2):
            for k_l in range(3):
                j_g = 3 + j_l
                k_g = k_l
                V_new[j_g * p + k_g, col] += v[j_l * 3 + k_l]
                
        for k_l in range(3):
            for i_l in range(2):
                k_g = k_l
                i_g = i_l
                W_new[k_g * n + i_g, col] += w[k_l * 2 + i_l]
                
        col += 1
        
    # Block (1,0): i in {2,3}, j in {0,1,2}, k in {3,4,5}
    # Local i in {0,1} -> Global i in {2,3}
    # Local k in {0,1,2} -> Global k in {3,4,5}
    for r in range(17):
        u = block_233_u[r]
        v = block_233_v[r]
        w = block_233_w[r]
        
        for i_l in range(2):
            for j_l in range(3):
                i_g = 2 + i_l
                j_g = j_l
                U_new[i_g * m + j_g, col] += u[i_l * 2 + j_l]
                
        for j_l in range(3):
            for k_l in range(3):
                j_g = j_l
                k_g = 3 + k_l
                V_new[j_g * p + k_g, col] += v[j_l * 3 + k_l]
                
        for k_l in range(3):
            for i_l in range(2):
                k_g = 3 + k_l
                i_g = 2 + i_l
                W_new[k_g * n + i_g, col] += w[k_l * 2 + i_l]
                
        col += 1
        
    # Block (1,1): i in {2,3}, j in {3,4}, k in {3,4,5}
    for r in range(11):
        u = block_223_u[r]
        v = block_223_v[r]
        w = block_223_w[r]
        
        for i_l in range(2):
            for j_l in range(2):
                i_g = 2 + i_l
                j_g = 3 + j_l
                U_new[i_g * m + j_g, col] += u[i_l * 2 + j_l]
                
        for j_l in range(2):
            for k_l in range(3):
                j_g = 3 + j_l
                k_g = 3 + k_l
                V_new[j_g * p + k_g, col] += v[j_l * 3 + k_l]
                
        for k_l in range(3):
            for i_l in range(2):
                k_g = 3 + k_l
                i_g = 2 + i_l
                W_new[k_g * n + i_g, col] += w[k_l * 2 + i_l]
                
        col += 1

    # Verify
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = len(U_arr[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if verify_decomp(U_new, V_new, W_new, n, m, p):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        return {"U": U_best, "V": V_best, "W": W_best}

# EVOLVE-BLOCK-END
