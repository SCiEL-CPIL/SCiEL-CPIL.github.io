"""
Rank-minimal exact tensor decomposition for 2x3 @ 3x7 matrix multiplication.

Strategy:
  - Partition the 2x3 matrix A into a 2x2 block (cols 0,1) and a 2x1 block (col 2).
  - Partition the 3x7 matrix B into a 2x7 block (rows 0,1) and a 1x7 block (row 2).
  - For the 2x2 @ 2x7 part, further partition the 2x7 into four 2x2 blocks and one 2x1 block.
    Each 2x2 @ 2x2 is handled with Strassen (rank 7), and 2x2 @ 2x1 with schoolbook (rank 4).
  - The 2x1 @ 1x7 part is handled with schoolbook (rank 14).
  - Total rank = 3 * 7 + 4 + 14 = 39, which is better than the schoolbook rank of 42.

The output W follows the convention W[k*n + i, r] where n=2, so W has shape (7*2, rank) = (14, rank).
"""

import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(offset_i, offset_j, offset_k, rank, U, V, W):
    """
    Add a Strassen decomposition for a 2x2 @ 2x2 sub-block to U, V, W.
    
    Parameters:
    - offset_i: starting row index in A (global)
    - offset_j: starting col index in A / row index in B (global)
    - offset_k: starting col index in B (global)
    - rank: current rank offset
    - U, V, W: factor matrices (lists of lists)
    
    Returns: new rank offset (rank + 7)
    """
    # Strassen 7 multiplications for 2x2 @ 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U indices: row = i*3 + j (n=2, m=3, so row index in A is i*m + j = i*3 + j)
    # V indices: row = j*7 + k (m=3, p=7, so row index in B is j*p + k = j*7 + k)
    # W indices: row = k*2 + i (p=7, n=2, so row index in C is k*n + i = k*2 + i)
    
    r = rank
    for _ in range(7):
        U.append([0] * 0)  # placeholder, will fix below
        V.append([0] * 0)
        W.append([0] * 0)
    
    # Actually, let's build the Strassen contribution properly.
    # We need to add 7 new columns to U, V, W.
    
    # For each Strassen multiplication, we add one column.
    
    # m1: (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    #   U col: A[0][0] and A[1][1] contribute +1
    #   V col: B[0][0] and B[1][1] contribute +1
    #   W col: C[0][0] += m1, C[0][1] += m1, C[1][0] += m1, C[1][1] += m1? 
    #   Actually: C[0][0] = m1 - m2 - m4 + m7, C[0][1] = m3 + m5, C[1][0] = m2 + m4, C[1][1] = m1 - m3 - m5 + m6
    
    # Let me be more careful. Standard Strassen:
    # C[0][0] = m1 - m2 - m4 + m7
    # C[0][1] = m3 + m5
    # C[1][0] = m2 + m4
    # C[1][1] = m1 - m3 - m5 + m6
    
    # m1 = (a+d)(e+h): 
    #   U: a=+1, d=+1 (rows 0 and 1 of the 2x2 A block, which correspond to global A rows offset_i and offset_i+1, but only col offset_j for a, col offset_j+1 for d)
    #   Wait: a = A[0][0] -> global row offset_i, col offset_j
    #         d = A[1][1] -> global row offset_i+1, col offset_j+1
    #   V: e = B[0][0] -> global row offset_j, col offset_k
    #         h = B[1][1] -> global row offset_j+1, col offset_k+1
    #   W: contributes to C[0][0] and C[1][1] with +1
    #       C[0][0] -> global row offset_k, col offset_i -> W index = offset_k*2 + offset_i
    #       C[1][1] -> global row offset_k+1, col offset_i+1 -> W index = (offset_k+1)*2 + (offset_i+1)
    
    # This is getting complex. Let me just build the 7 Strassen multiplications explicitly.
    
    # I'll remove the placeholder columns and do it properly.
    del U[-7:]
    del V[-7:]
    del W[-7:]
    
    n_local = 2
    m_local = 2
    p_local = 2
    
    # Define the 7 Strassen multipliers as (U_coeff, V_coeff, W_coeff)
    # Each is a triple of sparse vectors for the 2x2, 2x2, 2x2 local blocks.
    
    # m1 = (a+d)(e+h)
    m1_U = {(offset_i + 0, offset_j + 0): 1, (offset_i + 1, offset_j + 1): 1}
    m1_V = {(offset_j + 0, offset_k + 0): 1, (offset_j + 1, offset_k + 1): 1}
    m1_W = {(offset_k + 0, offset_i + 0): 1, (offset_k + 1, offset_i + 1): 1}
    
    # m2 = (c+d)e
    m2_U = {(offset_i + 1, offset_j + 0): 1, (offset_i + 1, offset_j + 1): 1}
    m2_V = {(offset_j + 0, offset_k + 0): 1}
    m2_W = {(offset_k + 0, offset_i + 1): 1}
    
    # m3 = a(f-h)
    m3_U = {(offset_i + 0, offset_j + 0): 1}
    m3_V = {(offset_j + 0, offset_k + 1): 1, (offset_j + 1, offset_k + 1): -1}
    m3_W = {(offset_k + 1, offset_i + 0): 1}
    
    # m4 = d(g-e)
    m4_U = {(offset_i + 1, offset_j + 1): 1}
    m4_V = {(offset_j + 1, offset_k + 0): 1, (offset_j + 0, offset_k + 0): -1}
    m4_W = {(offset_k + 0, offset_i + 1): 1}
    
    # m5 = (a+b)h
    m5_U = {(offset_i + 0, offset_j + 0): 1, (offset_i + 0, offset_j + 1): 1}
    m5_V = {(offset_j + 1, offset_k + 1): 1}
    m5_W = {(offset_k + 1, offset_i + 0): 1}
    
    # m6 = (c-a)(e+f)
    m6_U = {(offset_i + 1, offset_j + 0): 1, (offset_i + 0, offset_j + 0): -1}
    m6_V = {(offset_j + 0, offset_k + 0): 1, (offset_j + 0, offset_k + 1): 1}
    m6_W = {(offset_k + 1, offset_i + 1): 1}
    
    # m7 = (b-d)(g+h)
    m7_U = {(offset_i + 0, offset_j + 1): 1, (offset_i + 1, offset_j + 1): -1}
    m7_V = {(offset_j + 1, offset_k + 0): 1, (offset_j + 1, offset_k + 1): 1}
    m7_W = {(offset_k + 0, offset_i + 0): 1}
    
    # Now, the W contributions need to be summed for each C entry.
    # C[0][0] = m1 - m2 - m4 + m7
    # C[0][1] = m3 + m5
    # C[1][0] = m2 + m4
    # C[1][1] = m1 - m3 - m5 + m6
    
    # So for W, we need to accumulate:
    # W[global_C_row_offset_k + 0, global_C_col_offset_i + 0] gets: +1 from m1, -1 from m2, -1 from m4, +1 from m7
    # W[global_C_row_offset_k + 1, global_C_col_offset_i + 0] gets: +1 from m3, +1 from m5
    # W[global_C_row_offset_k + 0, global_C_col_offset_i + 1] gets: +1 from m2, +1 from m4
    # W[global_C_row_offset_k + 1, global_C_col_offset_i + 1] gets: +1 from m1, -1 from m3, -1 from m5, +1 from m6
    
    # Let's build the W contributions properly by summing.
    w_contrib = {}
    for mW, sign in [(m1_W, 1), (m2_W, -1), (m4_W, -1), (m7_W, 1)]:
        for (k, i), v in mW.items():
            key = (k, i)
            w_contrib[key] = w_contrib.get(key, 0) + sign * v
    for mW, sign in [(m3_W, 1), (m5_W, 1)]:
        for (k, i), v in mW.items():
            key = (k, i)
            w_contrib[key] = w_contrib.get(key, 0) + sign * v
    for mW, sign in [(m2_W, 1), (m4_W, 1)]:
        for (k, i), v in mW.items():
            key = (k, i)
            w_contrib[key] = w_contrib.get(key, 0) + sign * v
    for mW, sign in [(m1_W, 1), (m3_W, -1), (m5_W, -1), (m6_W, 1)]:
        for (k, i), v in mW.items():
            key = (k, i)
            w_contrib[key] = w_contrib.get(key, 0) + sign * v
    
    # Now build the 7 new columns
    U_new = []
    V_new = []
    W_new = []
    
    all_Us = [m1_U, m2_U, m3_U, m4_U, m5_U, m6_U, m7_U]
    all_Vs = [m1_V, m2_V, m3_V, m4_V, m5_V, m6_V, m7_V]
    
    for c in range(7):
        u_col = {}
        for (i, j), v in all_Us[c].items():
            u_col[(i, j)] = u_col.get((i, j), 0) + v
        v_col = {}
        for (j, k), v in all_Vs[c].items():
            v_col[(j, k)] = v_col.get((j, k), 0) + v
    
    # Actually, let me just directly add columns to U, V, W.
    n_global = 2
    m_global = 3
    p_global = 7
    
    for c in range(7):
        U_col = [0] * (n_global * m_global)  # 6 rows
        V_col = [0] * (m_global * p_global)  # 21 rows
        W_col = [0] * (p_global * n_global)  # 14 rows
        
        # Fill U_col
        for (i, j), v in all_Us[c].items():
            U_col[i * m_global + j] += v
        
        # Fill V_col
        for (j, k), v in all_Vs[c].items():
            V_col[j * p_global + k] += v
        
        # Fill W_col
        for (k, i), v in w_contrib.items():
            # Only include if this column's multiplier actually contributes
            # But w_contrib is the sum, not per-column. I need per-column W.
            pass
    
    # I realize I'm overcomplicating this. Let me just directly construct the 7 columns.
    del U_new, V_new, W_new
    
    # Simpler approach: for each of the 7 Strassen multipliers, add a column to U, V, W
    # where U and V are the factor matrices for the input, and W is the output coefficient.
    
    # For m1: U has +1 at (a) and (d), V has +1 at (e) and (h), W has +1 at C[0][0] and C[1][1]
    # But W should reflect the actual contribution to each C entry from this multiplier.
    
    # Let me restructure: for each multiplier r in 1..7, define:
    #   U_r: 6-dim vector
    #   V_r: 21-dim vector  
    #   W_r: 14-dim vector
    # such that sum_r U_r * V_r * W_r = tensor
    
    # m1:
    U_r1 = [0]*6
    U_r1[0*3+0] = 1  # a
    U_r1[1*3+1] = 1  # d
    V_r1 = [0]*21
    V_r1[0*7+0] = 1  # e
    V_r1[1*7+1] = 1  # h
    W_r1 = [0]*14
    W_r1[0*2+0] = 1  # C[0][0]
    W_r1[1*2+1] = 1  # C[1][1]
    
    # m2:
    U_r2 = [0]*6
    U_r2[1*3+0] = 1  # c
    U_r2[1*3+1] = 1  # d
    V_r2 = [0]*21
    V_r2[0*7+0] = 1  # e
    W_r2 = [0]*14
    W_r2[0*2+1] = 1  # C[1][0]
    
    # m3:
    U_r3 = [0]*6
    U_r3[0*3+0] = 1  # a
    V_r3 = [0]*21
    V_r3[0*7+1] = 1  # f
    V_r3[1*7+1] = -1  # h
    W_r3 = [0]*14
    W_r3[1*2+0] = 1  # C[0][1]
    
    # m4:
    U_r4 = [0]*6
    U_r4[1*3+1] = 1  # d
    V_r4 = [0]*21
    V_r4[1*7+0] = 1  # g
    V_r4[0*7+0] = -1  # e
    W_r4 = [0]*14
    W_r4[0*2+1] = 1  # C[1][0]
    
    # m5:
    U_r5 = [0]*6
    U_r5[0*3+0] = 1  # a
    U_r5[0*3+1] = 1  # b
    V_r5 = [0]*21
    V_r5[1*7+1] = 1  # h
    W_r5 = [0]*14
    W_r5[1*2+0] = 1  # C[0][1]
    
    # m6:
    U_r6 = [0]*6
    U_r6[1*3+0] = 1  # c
    U_r6[0*3+0] = -1  # a
    V_r6 = [0]*21
    V_r6[0*7+0] = 1  # e
    V_r6[0*7+1] = 1  # f
    W_r6 = [0]*14
    W_r6[1*2+1] = 1  # C[1][1]
    
    # m7:
    U_r7 = [0]*6
    U_r7[0*3+1] = 1  # b
    U_r7[1*3+1] = -1  # d
    V_r7 = [0]*21
    V_r7[1*7+0] = 1  # g
    V_r7[1*7+1] = 1  # h
    W_r7 = [0]*14
    W_r7[0*2+0] = 1  # C[0][0]
    
    # Now, the W vectors above are NOT correct because they don't account for the combination.
    # C[0][0] = m1 - m2 - m4 + m7, so W for C[0][0] should be:
    #   from m1: +1, from m2: -1, from m4: -1, from m7: +1
    # But in the tensor decomposition, each rank-1 term is U_r * V_r * W_r.
    # So W_r1[C[0][0]] = 1, W_r2[C[0][0]] = -1, W_r4[C[0][0]] = -1, W_r7[C[0][0]] = 1
    # Similarly for other entries.
    
    # Let me redo W properly.
    W_r1 = [0]*14
    W_r1[0*2+0] = 1   # C[0][0]
    W_r1[1*2+1] = 1   # C[1][1]
    
    W_r2 = [0]*14
    W_r2[0*2+1] = -1  # C[1][0] gets -1 from m2? No: C[1][0] = m2 + m4, so m2 contributes +1
    # Wait: C[0][0] = m1 - m2 - m4 + m7
    # So m2 contributes -1 to C[0][0], and +1 to C[1][0]
    W_r2[0*2+0] = -1  # C[0][0]
    W_r2[0*2+1] = 1   # C[1][0]
    
    W_r3 = [0]*14
    W_r3[1*2+0] = 1   # C[0][1]
    
    W_r4 = [0]*14
    W_r4[0*2+0] = -1  # C[0][0]
    W_r4[0*2+1] = 1   # C[1][0]
    
    W_r5 = [0]*14
    W_r5[1*2+0] = 1   # C[0][1]
    
    W_r6 = [0]*14
    W_r6[1*2+1] = 1   # C[1][1]
    
    W_r7 = [0]*14
    W_r7[0*2+0] = 1   # C[0][0]
    
    # Now add these 7 columns to U, V, W
    U.extend([U_r1, U_r2, U_r3, U_r4, U_r5, U_r6, U_r7])
    V.extend([V_r1, V_r2, V_r3, V_r4, V_r5, V_r6, V_r7])
    W.extend([W_r1, W_r2, W_r3, W_r4, W_r5, W_r6, W_r7])
    
    return rank + 7


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    U = []
    V = []
    W = []
    rank = 0
    
    # Block 1: A_1 (2x2) @ B_1 (2x7)
    # Partition B_1 into 2x2, 2x2, 2x2, 2x1 blocks
    # B_1 cols 0-1: 2x2, cols 2-3: 2x2, cols 4-5: 2x2, col 6: 2x1
    
    # Sub-block 1.1: A_1 @ B_11 (2x2 @ 2x2) using Strassen
    rank = _strassen_2x2x2(0, 0, 0, rank, U, V, W)
    
    if time.time() > deadline:
        break
    
    # Sub-block 1.2: A_1 @ B_12 (2x2 @ 2x2) using Strassen
    rank = _strassen_2x2x2(0, 0, 2, rank, U, V, W)
    
    if time.time() > deadline:
        break
    
    # Sub-block 1.3: A_1 @ B_13 (2x2 @ 2x2) using Strassen
    rank = _strassen_2x2x2(0, 0, 4, rank, U, V, W)
    
    if time.time() > deadline:
        break
    
    # Sub-block 1.4: A_1 @ B_14 (2x2 @ 2x1) using schoolbook (rank 4)
    # A_1 is 2x2, B_14 is 2x1
    # C_14 is 2x1
    # Schoolbook: 4 multiplications
    for i in range(2):
        for j in range(2):
            k = 6  # col 6 of B
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            u_col[i * m + j] = 1
            v_col[j * p + k] = 1
            w_col[k * n + i] = 1
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            rank += 1
    
    if time.time() > deadline:
        break
    
    # Block 2: A_2 (2x1) @ B_2 (1x7) using schoolbook (rank 14)
    # A_2 is col 2 of A (2x1), B_2 is row 2 of B (1x7)
    # C_2 is 2x7
    for i in range(2):
        for k in range(7):
            j = 2  # row/col index
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            u_col[i * m + j] = 1
            v_col[j * p + k] = 1
            w_col[k * n + i] = 1
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            rank += 1
    
    # Final rank should be 3*7 + 4 + 14 = 39
    # But let's verify: 3 Strassen blocks (7 each) = 21, + 4 schoolbook for 2x2x1 = 25, + 14 schoolbook for 2x1x7 = 39
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
