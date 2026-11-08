import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=6
    # U rows: 8
    # V rows: 24
    # W rows: 12
    
    # Strassen decomposition for 2x2x2
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # 7 multiplications
    # m1: (a00+a11)(b00+b11) -> c00+1, c11+1
    # m2: (a10+a11)b00 -> c10+1
    # m3: a00(b01-b11) -> c01+1
    # m4: a11(b10-b00) -> c00+1, c10+1
    # m5: (a00+a01)b11 -> c00-1, c01+1
    # m6: (a10-a00)(b00+b01) -> c11+1
    # m7: (a01-a11)(b10+b11) -> c00+1
    
    # U_local (4x7), V_local (4x7), W_local (4x7)
    U_loc = np.zeros((4, 7), dtype=int)
    V_loc = np.zeros((4, 7), dtype=int)
    W_loc = np.zeros((4, 7), dtype=int)
    
    # m1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # m2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1
    
    # m3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1
    
    # m4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # m5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # m6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # m7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Total blocks: 2 (j-blocks) * 3 (k-blocks) = 6
    num_j_blocks = 2
    num_k_blocks = 3
    total_rank = 6 * 7
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    for jb in range(num_j_blocks):
        for kb in range(num_k_blocks):
            for s in range(7):
                # Map local indices to global indices
                # A: local (r, c) -> global i=r, j=jb*2+c
                # B: local (r, c) -> global j=jb*2+r, k=kb*2+c
                # C: local (r, c) -> global k=kb*2+c, i=r
                
                for lr in range(2): # local row
                    for lc in range(2): # local col
                        l_idx = lr * 2 + lc
                        u_val = U_loc[l_idx, s]
                        v_val = V_loc[l_idx, s]
                        w_val = W_loc[l_idx, s]
                        
                        if u_val != 0:
                            g_i = lr
                            g_j = jb * 2 + lc
                            g_u_idx = g_i * m + g_j
                            U[g_u_idx, col_offset] += u_val
                        
                        if v_val != 0:
                            g_j = jb * 2 + lr
                            g_k = kb * 2 + lc
                            g_v_idx = g_j * p + g_k
                            V[g_v_idx, col_offset] += v_val
                        
                        if w_val != 0:
                            g_k = kb * 2 + lc
                            g_i = lr
                            g_w_idx = g_k * n + g_i
                            W[g_w_idx, col_offset] += w_val
                
                col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
