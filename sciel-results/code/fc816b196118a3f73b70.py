import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    t0 = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    # Fallback
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

    # Try Rank 34 (2x2x2 + 2x2x3)
    total_rank = 34
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Standard Strassen
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2; j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx // 2; k_l = idx % 2
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx // 2; k_l = idx % 2
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            col_off += 1
        return col_off

    def add_2x2x3_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Rank 10 decomposition for 2x2x3
        # First 7 terms: Strassen for k=0,1
        col_off = add_2x2x2_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off)
        
        # Next 3 terms for k=2
        # We need to compute C_i2 = A_i0 B_02 + A_i1 B_12
        # We add 3 terms to cover this.
        # This specific 10-term decomposition is non-trivial.
        # If we just add 4 terms, rank is 11.
        # I will fallback to 36 if I can't guarantee 10.
        # For now, I'll add 4 terms and let the rank be 38? No, that's worse.
        # I will return the 36 solution.
        return col_off

    # Since I don't have the verified rank 10 patterns, I will stick to the known 36 solution
    # but optimize the code structure.
    
    # Re-implement the 36 solution cleanly
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block_v2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2; j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx // 2; k_l = idx % 2
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx // 2; k_l = idx % 2
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            col_off += 1
        return col_off

    def add_2x2x1_block_v2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2; j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx; k_l = 0
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx; k_l = 0
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            col_off += 1
        return col_off

    col = 0
    # Block 1: A1 (0,1) * B1 (0,1) -> C1 (0,1)
    col = add_2x2x2_block_v2(U, V, W, 0, 0, 0, 0, 0, 0, col)
    # Block 2: A1 (0,1) * B2 (0,2) -> C2 (0,2)
    col = add_2x2x2_block_v2(U, V, W, 0, 0, 0, 2, 0, 2, col)
    # Block 3: A1 (0,1) * B3 (0,4) -> C3 (0,4)
    col = add_2x2x1_block_v2(U, V, W, 0, 0, 0, 4, 0, 4, col)
    # Block 4: A2 (0,2) * B1 (2,0) -> C4 (0,0)
    col = add_2x2x2_block_v2(U, V, W, 0, 2, 2, 0, 0, 0, col)
    # Block 5: A2 (0,2) * B2 (2,2) -> C5 (0,2)
    col = add_2x2x2_block_v2(U, V, W, 0, 2, 2, 2, 0, 2, col)
    # Block 6: A2 (0,2) * B3 (2,4) -> C6 (0,4)
    col = add_2x2x1_block_v2(U, V, W, 0, 2, 2, 4, 0, 4, col)

    assert col == total_rank

    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                if a % m != b // p: continue
                for c in range(pn):
                    if a % m == b // p and b % p == c // n and a // m == c % n: continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
