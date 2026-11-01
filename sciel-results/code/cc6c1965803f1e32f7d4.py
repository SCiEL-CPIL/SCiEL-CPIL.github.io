import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 100.0

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Attempt to construct rank 22 decomposition
    # Structure: 
    # 1. A[0:2, 0:2] * B[0:2, 0:2] (Strassen, rank 7)
    # 2. A[0:2, 0:2] * B[0:2, 2:4] (Strassen, rank 7)
    # 3. A[0:2, 2:3] * B[2:3, 0:4] (Schoolbook, rank 8)
    
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # Local indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
    # U, V, W are (4, 7)
    strassen_U = np.zeros((4, 7))
    strassen_V = np.zeros((4, 7))
    strassen_W = np.zeros((4, 7))
    
    # Term 1: (a+d)(e+h)
    strassen_U[0, 0] = 1; strassen_U[3, 0] = 1
    strassen_V[0, 0] = 1; strassen_V[3, 0] = 1
    strassen_W[0, 0] = 1; strassen_W[3, 0] = 1
    
    # Term 2: (c+d)e
    strassen_U[2, 1] = 1; strassen_U[3, 1] = 1
    strassen_V[0, 1] = 1
    strassen_W[2, 1] = 1
    
    # Term 3: a(f-h)
    strassen_U[0, 2] = 1
    strassen_V[1, 2] = 1; strassen_V[3, 2] = -1
    strassen_W[2, 2] = 1
    
    # Term 4: d(g-e)
    strassen_U[3, 3] = 1
    strassen_V[2, 3] = 1; strassen_V[0, 3] = -1
    strassen_W[0, 3] = 1; strassen_W[2, 3] = 1
    
    # Term 5: (a+b)h
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_V[3, 4] = 1
    strassen_W[0, 4] = -1; strassen_W[2, 4] = 1
    
    # Term 6: (c-a)(e+f)
    strassen_U[2, 5] = 1; strassen_U[0, 5] = -1
    strassen_V[0, 5] = 1; strassen_V[1, 5] = 1
    strassen_W[3, 5] = 1
    
    # Term 7: (b-d)(g+h)
    strassen_U[1, 6] = 1; strassen_U[3, 6] = -1
    strassen_V[2, 6] = 1; strassen_V[3, 6] = 1
    strassen_W[0, 6] = 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # Global U index: i*3 + j (i in 0,1; j in 0,1)
    # Global V index: j*4 + k (j in 0,1; k in 0,1)
    # Global W index: k*2 + i (k in 0,1; i in 0,1)
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 2 + j
                g_u_idx = i * 3 + j
                U[g_u_idx, r] = strassen_U[u_idx, r]
                
                v_idx = j * 2 + (0) # k is 0,1 in local, but we need to map all k
                # Actually, strassen_V is indexed by j*2+k.
                # We need to iterate over j,k in 0,1
                # Let's just copy the whole block with index mapping
                
    # Redo Block 1 mapping explicitly
    U[:, :7] = 0
    V[:, :7] = 0
    W[:, :7] = 0
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                # U part
                loc_u = i * 2 + j
                g_u = i * 3 + j
                U[g_u, r] = strassen_U[loc_u, r]
                
                # V part (j in 0,1; k in 0,1)
                for k in range(2):
                    loc_v = j * 2 + k
                    g_v = j * 4 + k
                    V[g_v, r] = strassen_V[loc_v, r]
                    
                # W part (k in 0,1; i in 0,1)
                for k in range(2):
                    loc_w = k * 2 + i
                    g_w = k * 2 + i
                    W[g_w, r] = strassen_W[loc_w, r]
                    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # Global U index: i*3 + j (i in 0,1; j in 0,1)
    # Global V index: j*4 + (k+2) (j in 0,1; k in 0,1)
    # Global W index: (k+2)*2 + i (k in 0,1; i in 0,1)
    
    for r in range(7, 14):
        r_loc = r - 7
        for i in range(2):
            for j in range(2):
                # U part
                loc_u = i * 2 + j
                g_u = i * 3 + j
                U[g_u, r] = strassen_U[loc_u, r_loc]
                
                # V part (j in 0,1; k in 0,1 -> global k in 2,3)
                for k in range(2):
                    loc_v = j * 2 + k
                    g_v = j * 4 + (k + 2)
                    V[g_v, r] = strassen_V[loc_v, r_loc]
                    
                # W part (k in 0,1; i in 0,1 -> global k in 2,3)
                for k in range(2):
                    loc_w = k * 2 + i
                    g_w = (k + 2) * 2 + i
                    W[g_w, r] = strassen_W[loc_w, r_loc]
                    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    # A index: i in 0,1; j=2. Global U: i*3 + 2.
    # B index: j=2; k in 0,1,2,3. Global V: 2*4 + k.
    # C index: i in 0,1; k in 0,1,2,3. Global W: k*2 + i.
    
    for r in range(14, 22):
        t = r - 14
        i = t // 4
        k = t % 4
        
        U[i * 3 + 2, r] = 1
        V[2 * 4 + k, r] = 1
        W[k * 2 + i, r] = 1
        
    # Verify
    def check_solution(U, V, W, r):
        for c in range(r):
            for val in U[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in V[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in W[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx_val] * V[c, v_idx_val] * W[c, w_idx_val]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if check_solution(U, V, W, rank):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}
