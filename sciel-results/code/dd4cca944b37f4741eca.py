import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin

    n, m, p = DIMENSIONS

    # Fallback: Schoolbook (Rank 32)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Hardcoded Rank-28 Decomposition
    # The 2x4x4 multiplication tensor decomposes into 4 independent 2x2x2 tensors.
    # We use the known Strassen rank-7 decomposition for each 2x2x2 block.
    
    # 1. Define the Strassen factors for 2x2x2
    # Indices for 2x2 matrices:
    # A: a(0), b(1), c(2), d(3)
    # B: e(0), f(1), g(2), h(3)
    # C: i(0), k(1), j(2), l(3)  (Note: C flat index is col*2 + row)
    # C[0,0]=i->0, C[1,0]=k->1, C[0,1]=j->2, C[1,1]=l->3
    
    # Verified Strassen factors for 2x2x2 (Integer entries)
    # Reference: Standard Strassen Algorithm
    # m1 = (a+d)(e+h)
    # m2 = (c+d)h
    # m3 = a(e+f)
    # m4 = d(g+f)
    # m5 = (a+b)h
    # m6 = (c+d)(g-h)
    # m7 = (a-b)(e+h)
    
    # Let's verify these produce the correct C.
    # C11 (i) = m1 - m4 + m5 - m7
    # C21 (k) = m2 + m6
    # C12 (j) = m3 + m4
    # C22 (l) = m1 - m2 + m3 - m5
    
    # Let's map these to U (A), V (B), W (C) vectors for the 7 terms.
    # Term 1: (a+d) * (e+h)
    # U1: [1, 0, 0, 1]
    # V1: [1, 0, 0, 1]
    # W1: [1, 0, 0, 0] (contributes to i)
    #      Wait, W must account for all outputs.
    # i = ... + 1*m1 ...
    # k = ... + 0*m1 ...
    # j = ... + 0*m1 ...
    # l = ... + 1*m1 ...
    # So W1 = [1, 0, 0, 1]
    
    # Term 2: (c+d) * h
    # U2: [0, 0, 1, 1]
    # V2: [0, 0, 0, 1]
    # k = ... + 1*m2 ...
    # l = ... - 1*m2 ...
    # W2 = [0, 1, 0, -1]
    
    # Term 3: a * (e+f)
    # U3: [1, 0, 0, 0]
    # V3: [1, 1, 0, 0]
    # j = ... + 1*m3 ...
    # l = ... + 1*m3 ...
    # W3 = [0, 0, 1, 1]
    
    # Term 4: d * (g+f)
    # U4: [0, 0, 0, 1]
    # V4: [0, 1, 1, 0]
    # i = ... - 1*m4 ...
    # j = ... + 1*m4 ...
    # W4 = [-1, 0, 1, 0]
    
    # Term 5: (a+b) * h
    # U5: [1, 1, 0, 0]
    # V5: [0, 0, 0, 1]
    # i = ... + 1*m5 ...
    # l = ... - 1*m5 ...
    # W5 = [1, 0, 0, -1]
    
    # Term 6: (c+d) * (g-h)
    # U6: [0, 0, 1, 1]
    # V6: [0, 0, 1, -1]
    # k = ... + 1*m6 ...
    # W6 = [0, 1, 0, 0]
    
    # Term 7: (a-b) * (e+h)
    # U7: [1, -1, 0, 0]
    # V7: [1, 0, 0, 1]
    # i = ... - 1*m7 ...
    # W7 = [-1, 0, 0, 0]
    
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # a
        [0, 0, 0, 0, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 1, 0]   # d
    ])
    
    V_s = np.array([
        [1, 0, 1, 0, 0, 0, 1],  # e
        [0, 0, 1, 1, 0, 0, 0],  # f
        [0, 0, 0, 1, 0, 1, 0],  # g
        [1, 1, 0, 0, 1, -1, 1]  # h
    ])
    
    W_s = np.array([
        [1, 0, 0, -1, 1, 0, -1], # i (C[0,0])
        [0, 1, 0, 0, 0, 1, 0],   # k (C[1,0])
        [0, 0, 1, 1, 0, 0, 0],   # j (C[0,1])
        [1, -1, 1, 0, -1, 0, 0]  # l (C[1,1])
    ])
    
    # Verify 2x2x2 decomposition
    T_2x2x2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_2x2x2[i*2+j, j*2+k, k*2+i] = 1
                
    # Check: sum_s U_s[i,s] V_s[j,s] W_s[k,s] == T_2x2x2[i,j,k]
    valid_2x2 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U_s[i, :] * V_s[j, :] * W_s[k, :])
                if abs(val - T_2x2x2[i, j, k]) > 1e-9:
                    valid_2x2 = False
                    break
            if not valid_2x2: break
        if not valid_2x2: break
        
    if not valid_2x2:
        # If hardcoded factors are wrong (they shouldn't be), fallback to schoolbook
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Construct full 2x4x4 decomposition (Rank 28)
    rank_28 = 28
    U_28 = np.zeros((8, rank_28), dtype=float)
    V_28 = np.zeros((16, rank_28), dtype=float)
    W_28 = np.zeros((8, rank_28), dtype=float)
    
    # Subproblems:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # 3. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    subproblems = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2)
    ]
    
    for sp_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(subproblems):
        for s in range(7):
            r = sp_idx * 7 + s
            
            # U: A factors
            # Local A (row, col) -> Global A (row, a_col_off + col)
            # Global A flat: row * 4 + col
            for local_row in range(2):
                for local_col in range(2):
                    global_col = a_col_off + local_col
                    i_flat = local_row * 4 + global_col
                    local_i = local_row * 2 + local_col
                    U_28[i_flat, r] = U_s[local_i, s]
                    
            # V: B factors
            # Local B (row, col) -> Global B (b_row_off + row, b_col_off + col)
            # Global B flat: row * 4 + col
            for local_row in range(2):
                for local_col in range(2):
                    global_row = b_row_off + local_row
                    global_col = b_col_off + local_col
                    j_flat = global_row * 4 + global_col
                    local_j = local_row * 2 + local_col
                    V_28[j_flat, r] = V_s[local_j, s]
                    
            # W: C factors
            # Local C (row, col) -> Global C (row, c_col_off + col)
            # Global C flat: col * 2 + row
            for local_row in range(2):
                for local_col in range(2):
                    global_col = c_col_off + local_col
                    k_flat = global_col * 2 + local_row
                    local_k = local_col * 2 + local_row
                    W_28[k_flat, r] = W_s[local_k, s]

    # Verify the full 2x4x4 decomposition
    T_full = np.zeros((8, 16, 8), dtype=float)
    for row_a in range(2):
        for col_a in range(4):
            i = row_a * 4 + col_a
            for row_b in range(4):
                for col_b in range(4):
                    j = row_b * 4 + col_b
                    for row_c in range(2):
                        for col_c in range(4):
                            k = col_c * 2 + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T_full[i, j, k] = 1
    
    max_error = 0.0
    # Efficient check using tensor contraction logic if possible, or just loops for 8x16x8 (1024 elements)
    # 1024 * 28 ops is trivial.
    for i in range(8):
        for j in range(16):
            for k in range(8):
                val = np.sum(U_28[i, :] * V_28[j, :] * W_28[k, :])
                err = abs(val - T_full[i, j, k])
                if err > max_error:
                    max_error = err
    
    if max_error < 1e-9:
        best_U, best_V, best_W = U_28, V_28, W_28
        best_rank = 28
        
    if time.time() - start_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
