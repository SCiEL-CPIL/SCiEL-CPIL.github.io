import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5
    # Rank 36
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Try Kronecker substitution / cyclic convolution approach
    # Embed 2x4x5 into cyclic convolution of length 8
    # A (2x4) -> polynomial of degree 7 (8 coefficients)
    # B (4x5) -> we need a different encoding
    # 
    # Alternative: use 4 separate length-8 cyclic convolutions
    # For each row j of B (j=0..3):
    #   A_0(x) = a_{0,0} + a_{0,1}x + a_{0,2}x^2 + a_{0,3}x^3 (degree 3)
    #   A_1(x) = a_{1,0} + a_{1,1}x + a_{1,2}x^2 + a_{1,3}x^3 (degree 3)
    #   B_j(x) = b_{j,0} + b_{j,1}x + b_{j,2}x^2 + b_{j,3}x^3 (degree 3)
    #   Product has degree 6, use cyclic convolution of length 7 or 8
    #
    # Winograd F(8) cyclic convolution has rank 7
    # But we need to extract specific coefficients
    #
    # Let's try: for each of the 4 rows of B, do a cyclic convolution of length 8
    # A_0(x) padded to degree 7: a_{0,0} + a_{0,1}x + a_{0,2}x^2 + a_{0,3}x^3
    # A_1(x) padded to degree 7: a_{1,0} + a_{1,1}x + a_{1,2}x^2 + a_{1,3}x^3
    # B_j(x) padded to degree 7: b_{j,0} + b_{j,1}x + b_{j,2}x^2 + b_{j,3}x^3
    #
    # (A_0 * B_j)[3] = a_{0,0}b_{j,3} + a_{0,1}b_{j,2} + a_{0,2}b_{j,1} + a_{0,3}b_{j,0} + wraparound terms
    # This doesn't directly give C_{0,j} = sum_k a_{0,k} b_{k,j}
    #
    # The Kronecker substitution doesn't cleanly separate the indices for this problem.
    # Let's try a different approach: direct rank-35 search via perturbation.
    
    # Try to find rank-35 by searching for a column that can be removed or merged
    # Strategy: for each pair of columns, check if their sum/difference preserves validity
    # with appropriate scaling of other columns
    
    found_rank35 = False
    U35 = None
    V35 = None
    W35 = None
    
    if time.time() < deadline - 20:
        # Try removing each column and checking if the remaining 35 columns still span
        # the same tensor (i.e., the removed column is a linear combination of others
        # in a way that preserves the bilinear structure)
        # This is complex; instead, try a simpler heuristic:
        # Check if any two columns have a special relationship that allows merging
        
        # Alternative: try a different block decomposition
        # What if we use a 2x3x3 block (rank 9) instead of 2x2x2+2x2x1 (rank 11)?
        # 2x3x3: A is 2x3, B is 3x3, C is 2x3
        # Schoolbook rank: 18, Strassen-like: ~12-13
        # Total: 2 blocks of 2x3x3 (rank ~12 each) + 2 blocks of 2x1x2 (rank 4 each) = 32?
        
        # Let's try: partition columns of B as (3,2) instead of (2,2,1)
        # Block 1: A[:, 0:3] (2x3) * B[0:3, 0:3] (3x3) -> C[0:2, 0:3] (2x3)
        # Block 2: A[:, 0:3] (2x3) * B[0:3, 3:5] (3x2) -> C[0:2, 3:5] (2x2)
        # Block 3: A[:, 3:4] (2x1) * B[3:4, 0:3] (1x3) -> C[0:2, 0:3] (2x3)
        # Block 4: A[:, 3:4] (2x1) * B[3:4, 3:5] (1x2) -> C[0:2, 3:5] (2x2)
        
        # This doesn't seem to help. Let's just verify and return rank 36.
        pass

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
