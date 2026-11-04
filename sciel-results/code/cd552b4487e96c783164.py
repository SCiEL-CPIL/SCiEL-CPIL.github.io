import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Row indices: for U: (i,j) -> i*2+j; for V: (j,k) -> j*2+k; for W: (k,i) -> k*2+i
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Recursive construction of 4x4 factors from 2x2 factors
    # U4[i, j, t] = U2[i_out, j_out, t_out] * U2[i_in, j_in, t_in]
    # where i = 2*i_out + i_in, j = 2*j_out + j_in
    # t = 7*t_out + t_in (Kronecker product of the rank dimension)
    
    # Initialize U4, V4, W4 with zeros
    # Shape: (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Fill U4
    # i, j range from 0 to 3
    for i in range(4):
        for j in range(4):
            i_out, i_in = i // 2, i % 2
            j_out, j_in = j // 2, j % 2
            # U2 rows are indexed by i*2+j
            row_u2_out = i_out * 2 + j_out
            row_u2_in = i_in * 2 + j_in
            
            # For each combination of t_out and t_in
            for t_out in range(7):
                u_val_out = U2[row_u2_out, t_out]
                if u_val_out == 0: continue
                for t_in in range(7):
                    u_val_in = U2[row_u2_in, t_in]
                    if u_val_in == 0: continue
                    t = t_out * 7 + t_in
                    U4[i * 4 + j, t] = u_val_out * u_val_in

    # Fill V4
    # j, k range from 0 to 3
    for j in range(4):
        for k in range(4):
            j_out, j_in = j // 2, j % 2
            k_out, k_in = k // 2, k % 2
            row_v2_out = j_out * 2 + k_out
            row_v2_in = j_in * 2 + k_in
            
            for t_out in range(7):
                v_val_out = V2[row_v2_out, t_out]
                if v_val_out == 0: continue
                for t_in in range(7):
                    v_val_in = V2[row_v2_in, t_in]
                    if v_val_in == 0: continue
                    t = t_out * 7 + t_in
                    V4[j * 4 + k, t] = v_val_out * v_val_in

    # Fill W4
    # k, i range from 0 to 3
    # Trace convention: W[k*n+i] -> W[k*4+i]
    for k in range(4):
        for i in range(4):
            k_out, k_in = k // 2, k % 2
            i_out, i_in = i // 2, i % 2
            # W2 row index is k*2 + i
            row_w2_out = k_out * 2 + i_out
            row_w2_in = k_in * 2 + i_in
            
            for t_out in range(7):
                w_val_out = W2[row_w2_out, t_out]
                if w_val_out == 0: continue
                for t_in in range(7):
                    w_val_in = W2[row_w2_in, t_in]
                    if w_val_in == 0: continue
                    t = t_out * 7 + t_in
                    W4[k * 4 + i, t] = w_val_out * w_val_in

    # Runtime verification
    rank = 49
    
    # Build index arrays for all (i,j,k) triples
    i_idx = np.repeat(np.arange(4), 16)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    U_rows = U4[i_idx * 4 + j_idx, :]
    V_rows = V4[j_idx * 4 + k_idx, :]
    W_rows = W4[k_idx * 4 + i_idx, :]
    
    products = U_rows * V_rows * W_rows
    totals = products.sum(axis=1)
    
    valid = np.all(totals == 1)
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
