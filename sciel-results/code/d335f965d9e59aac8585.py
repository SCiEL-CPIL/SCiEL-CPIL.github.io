"""Program to find a rank-minimal tensor decomposition for 3x5x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strassen 2x2x2 decomposition
    strassen_u = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    strassen_w = [
        [1, 0, 0, 1], # C11=1, C22=1
        [0, 0, 1, -1],# C21=1, C22=-1
        [0, 1, 0, 1], # C12=1, C22=1
        [1, 0, 1, 0], # C11=1, C21=1
        [-1, 1, 0, 0],# C11=-1, C12=1
        [0, 0, 0, 1], # C22=1
        [1, 0, 0, 1]  # C11=1, C22=1
    ]
    
    # Define blocks: (i_start, j_start, k_start)
    # We can fit four 2x2x2 blocks in the 3x5x5 space
    # i: 0,1 (size 2)
    # j: 0,1 and 2,3 (size 2 each)
    # k: 0,1 and 2,3 (size 2 each)
    blocks = [
        (0, 0, 0),
        (0, 2, 2),
        (0, 0, 2),
        (0, 2, 0)
    ]
    
    # Determine which (i,j,k) are covered by Strassen blocks
    covered = np.zeros((n, m, p), dtype=bool)
    for i_s, j_s, k_s in blocks:
        for i in range(i_s, i_s+2):
            for j in range(j_s, j_s+2):
                for k in range(k_s, k_s+2):
                    covered[i, j, k] = True
    
    # Count rank
    rank_strassen = 7 * len(blocks)
    rank_schoolbook = np.sum(~covered)
    total_rank = rank_strassen + rank_schoolbook
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Add Strassen blocks
    for i_s, j_s, k_s in blocks:
        for r in range(7):
            u_r = strassen_u[r]
            v_r = strassen_v[r]
            w_r = strassen_w[r]
            
            for li in range(2):
                for lj in range(2):
                    a_idx = (i_s + li) * m + (j_s + lj)
                    U_full[a_idx, col_offset] = u_r[li * 2 + lj]
                    
                    for lk in range(2):
                        b_idx = (j_s + lj) * p + (k_s + lk)
                        V_full[b_idx, col_offset] = v_r[lj * 2 + lk]
                        
                        c_idx = (k_s + lk) * n + (i_s + li)
                        W_full[c_idx, col_offset] = w_r[li * 2 + lk]
            col_offset += 1
            
    # Add Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                max_err = max(max_err, abs(val - 1.0))
                
    if max_err < 1e-6:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
