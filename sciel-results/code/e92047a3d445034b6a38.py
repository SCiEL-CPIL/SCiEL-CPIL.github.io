import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 115.0

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

    # Pre-compute target tensor for fast verification
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition_fast(U, V, W):
        # Use einsum for fast tensor reconstruction and comparison
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Result: (nm, mp, pn)
        T = np.einsum('ir, jr, kr -> ijk', U, V, W)
        return np.allclose(T, target, atol=1e-9)

    # Verify rank-36 first
    if not verify_decomposition_fast(U, V, W):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        return best_result

    # Search for rank-35
    found_rank35 = False
    U35 = None
    V35 = None
    W35 = None

    # Extended set of alpha values for column merging
    alphas = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    # Try all pairs of columns with extended alpha set
    for i_col in range(total_rank):
        if time.time() > deadline - 15:
            break
        for j_col in range(i_col + 1, total_rank):
            if time.time() > deadline - 10:
                break
            for alpha in alphas:
                if time.time() > deadline - 5:
                    break
                # Create copies
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()

                # Merge column j_col into i_col: U_i += alpha * U_j
                U_test[:, i_col] += alpha * U_test[:, j_col]
                V_test[:, i_col] += alpha * V_test[:, j_col]
                W_test[:, i_col] += alpha * W_test[:, j_col]

                # Remove column j_col
                U_test = np.delete(U_test, j_col, axis=1)
                V_test = np.delete(V_test, j_col, axis=1)
                W_test = np.delete(W_test, j_col, axis=1)

                if verify_decomposition_fast(U_test, V_test, W_test):
                    found_rank35 = True
                    U35 = U_test
                    V35 = V_test
                    W35 = W_test
                    break
            if found_rank35 or time.time() > deadline - 10:
                break
        if found_rank35 or time.time() > deadline - 15:
            break

    if found_rank35:
        # Round to nearest half-integer
        U35_list = [[int(round(U35[i, r] * 2)) / 2 for r in range(35)] for i in range(nm)]
        V35_list = [[int(round(V35[i, r] * 2)) / 2 for r in range(35)] for i in range(mp)]
        W35_list = [[int(round(W35[i, r] * 2)) / 2 for r in range(35)] for i in range(pn)]
        return {"U": U35_list, "V": V35_list, "W": W35_list}

    # If rank 35 not found, return rank 36
    U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
    V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
    W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
