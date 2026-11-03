import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build a 5×5×5 decomposition and truncate to 3×5×5.
    # Partition 5 = 2+2+1.
    # Block sizes: [2, 2, 1]
    
    # For the 5×5×5 problem:
    # A is 5×5, B is 5×5, C is 5×5
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We'll build U, V, W for the 5×5×5 case, then truncate.
    
    # U has 25 rows (i,j) for i,j in 0..4
    # V has 25 rows (j,k) for j,k in 0..4
    # W has 25 rows (k,i) for k,i in 0..4
    
    # For each block combination, add rank-1 terms.
    
    block_sizes = [2, 2, 1]
    block_offsets = [0, 2, 4]
    
    U_555 = {}  # (i,j) -> list of (col, val)
    V_555 = {}  # (j,k) -> list of (col, val)
    W_555 = {}  # (k,i) -> list of (col, val)
    
    col = 0
    
    for a_row_blk in range(3):
        for a_col_blk in range(3):
            for b_col_blk in range(3):
                # a_col_blk = b_row_blk
                na = block_sizes[a_row_blk]
                ma = block_sizes[a_col_blk]
                mb = block_sizes[a_col_blk]  # same as ma
                pb = block_sizes[b_col_blk]
                
                # Subproblem: na × ma * mb × pb
                # A_sub[i,j] for i in 0..na-1, j in 0..ma-1
                # B_sub[j,k] for j in 0..mb-1, k in 0..pb-1
                # C_sub[i,k] for i in 0..na-1, k in 0..pb-1
                
                if na == 2 and ma == 2 and pb == 2:
                    # Use Strassen, rank 7
                    # Local indices: A[li,lj], B[lj,lk], C[li,lk] for li,lj,lk in 0,1
                    # Global: i = block_offsets[a_row_blk] + li, j = block_offsets[a_col_blk] + lj, k = block_offsets[b_col_blk] + lk
                    # Strassen terms:
                    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
                    # M2 = (A[1,0]+A[1,1])B[0,0]
                    # M3 = A[0,0](B[0,1]-B[1,1])
                    # M4 = A[1,1](B[1,0]-B[0,0])
                    # M5 = (A[0,0]+A[0,1])B[1,1]
                    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
                    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
                    
                    # M1
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+0), []).append((col, 1))
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+1), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+0), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+1), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+0), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+1), []).append((col, 1))
                    
                    # M2
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+0), []).append((col, 1))
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+1), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+0), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+1), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+1), []).append((col, -1))
                    
                    # M3
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+0), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+1), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+1), []).append((col, -1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+0), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+1), []).append((col, 1))
                    
                    # M4
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+1), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+0), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+0), []).append((col, -1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+0), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+1), []).append((col, 1))
                    
                    # M5
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+0), []).append((col, 1))
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+1), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+1), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+0), []).append((col, -1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+0), []).append((col, 1))
                    
                    # M6
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+0), []).append((col, 1))
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+0), []).append((col, -1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+0), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+0, block_offsets[b_col_blk]+1), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+1, block_offsets[a_row_blk]+1), []).append((col, 1))
                    
                    # M7
                    col += 1
                    U_555.setdefault((block_offsets[a_row_blk]+0, block_offsets[a_col_blk]+1), []).append((col, 1))
                    U_555.setdefault((block_offsets[a_row_blk]+1, block_offsets[a_col_blk]+1), []).append((col, -1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+0), []).append((col, 1))
                    V_555.setdefault((block_offsets[a_col_blk]+1, block_offsets[b_col_blk]+1), []).append((col, 1))
                    W_555.setdefault((block_offsets[b_col_blk]+0, block_offsets[a_row_blk]+0), []).append((col, 1))
                
                else:
                    # Schoolbook: rank = na * ma * pb
                    for li in range(na):
                        for lj in range(ma):
                            for lk in range(pb):
                                col += 1
                                i = block_offsets[a_row_blk] + li
                                j = block_offsets[a_col_blk] + lj
                                k = block_offsets[b_col_blk] + lk
                                U_555.setdefault((i, j), []).append((col, 1))
                                V_555.setdefault((j, k), []).append((col, 1))
                                W_555.setdefault((k, i), []).append((col, 1))
    
    # Now truncate to 3×5×5: keep only rows where i < 3 in U and W.
    # U: (i,j) where i in 0..2 (3 rows) and j in 0..4 (5 cols) → 15 rows
    # V: (j,k) where j in 0..4, k in 0..4 → 25 rows (unchanged)
    # W: (k,i) where k in 0..4, i in 0..2 → 15 rows
    
    rank = col
    U = [[0] * rank for _ in range(3 * 5)]
    V = [[0] * rank for _ in range(5 * 5)]
    W = [[0] * rank for _ in range(5 * 3)]
    
    for (i, j), entries in U_555.items():
        if i < 3:
            row = i * 5 + j
            for col_idx, val in entries:
                U[row][col_idx - 1] = val
    
    for (j, k), entries in V_555.items():
        row = j * 5 + k
        for col_idx, val in entries:
            V[row][col_idx - 1] = val
    
    for (k, i), entries in W_555.items():
        if i < 3:
            row = k * 3 + i
            for col_idx, val in entries:
                W[row][col_idx - 1] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
