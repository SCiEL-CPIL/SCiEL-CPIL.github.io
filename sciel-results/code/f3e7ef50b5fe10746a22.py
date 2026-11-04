"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]

    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]

        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]

    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True


def _make_1x2x4():
    # Rank 8 schoolbook for 1x2 matrix times 2x4 matrix
    u = np.zeros((1 * 2, 8))
    v = np.zeros((2 * 4, 8))
    w = np.zeros((4 * 1, 8))
    idx = 0
    for j in range(2):
        for k in range(4):
            u[j, idx] = 1.0
            v[j * 4 + k, idx] = 1.0
            w[k * 1, idx] = 1.0
            idx += 1
    return u, v, w


def _make_2x2x4_strassen():
    # Rank 14 for 2x2 x 2x4 using two Strassen 2x2x2 blocks
    u_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T

    v_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T

    w_s = np.zeros((4, 7))
    w_s[0, 0] = 1; w_s[3, 0] = 1
    w_s[1, 1] = 1; w_s[3, 1] = -1
    w_s[2, 2] = 1; w_s[3, 2] = 1
    w_s[0, 3] = 1; w_s[1, 3] = 1
    w_s[0, 4] = -1; w_s[2, 4] = 1
    w_s[3, 5] = 1
    w_s[0, 6] = 1

    u_224 = np.zeros((4, 14))
    v_224 = np.zeros((8, 14))
    w_224 = np.zeros((8, 14))

    for c in range(7):
        u_224[:, c] = u_s[:, c]
        v_224[0, c] = v_s[0, c]
        v_224[1, c] = v_s[1, c]
        v_224[4, c] = v_s[2, c]
        v_224[5, c] = v_s[3, c]
        w_224[0, c] = w_s[0, c]
        w_224[1, c] = w_s[1, c]
        w_224[2, c] = w_s[2, c]
        w_224[3, c] = w_s[3, c]

        u_224[:, c + 7] = u_s[:, c]
        v_224[2, c + 7] = v_s[0, c]
        v_224[3, c + 7] = v_s[1, c]
        v_224[6, c + 7] = v_s[2, c]
        v_224[7, c + 7] = v_s[3, c]
        w_224[4, c + 7] = w_s[0, c]
        w_224[5, c + 7] = w_s[1, c]
        w_224[6, c + 7] = w_s[2, c]
        w_224[7, c + 7] = w_s[3, c]

    return u_224, v_224, w_224


def _embed_subblock(i_off, j_off, k_off, sub_n, sub_m, sub_p,
                    u_sub, v_sub, w_sub, n=3, m=4, p=8):
    """
    Embed a sub-block decomposition into the full 3x4x8 space.
    Returns three columns (u_col, v_col, w_col) for one rank-1 slice.
    """
    rank_sub = u_sub.shape[1]
    cols_u = []
    cols_v = []
    cols_w = []

    for c in range(rank_sub):
        u_col = np.zeros(n * m)
        v_col = np.zeros(m * p)
        w_col = np.zeros(p * n)

        for i in range(sub_n):
            for j in range(sub_m):
                u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]

        for j in range(sub_m):
            for k in range(sub_p):
                v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]

        for k in range(sub_p):
            for i in range(sub_n):
                w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]

        cols_u.append(u_col)
        cols_v.append(v_col)
        cols_w.append(w_col)

    return (np.array(cols_u).T if cols_u else np.zeros((n * m, 0)),
            np.array(cols_v).T if cols_v else np.zeros((m * p, 0)),
            np.array(cols_w).T if cols_w else np.zeros((p * n, 0)))


def _construct_rank_88():
    """
    Construct a rank-88 decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy:
    - Split A (3x4) into A1 (1x4) and A2 (2x4)
    - Split B (4x8) into B1 (4x4) and B2 (4x4)
    - For A1 x Bj: split the 1x4 into two 1x2 blocks, each rank 8 -> 16 per Bj
    - For A2 x Bj: split the 2x4 into two 2x2 blocks, each rank 14 -> 28 per Bj
    Total: 16 + 16 + 28 + 28 = 88
    """
    n, m, p = 3, 4, 8

    u_124, v_124, w_124 = _make_1x2x4()
    u_224, v_224, w_224 = _make_2x2x4_strassen()

    U_parts = []
    V_parts = []
    W_parts = []

    # A1 (row 0) x B1 (cols 0-3)
    u1, v1, w1 = _embed_subblock(0, 0, 0, 1, 2, 4, u_124, v_124, w_124, n, m, p)
    U_parts.append(u1); V_parts.append(v1); W_parts.append(w1)

    # A1 (row 0) x B1 (cols 2-3) -- wait, B1 is 4x4, split into two 2-column halves
    # Actually, A1 is 1x4, so j_off goes 0 and 2
    u2, v2, w2 = _embed_subblock(0, 2, 0, 1, 2, 4, u_124, v_124, w_124, n, m, p)
    U_parts.append(u2); V_parts.append(v2); W_parts.append(w2)

    # A1 (row 0) x B2 (cols 4-7), first half
    u3, v3, w3 = _embed_subblock(0, 0, 4, 1, 2, 4, u_124, v_124, w_124, n, m, p)
    U_parts.append(u3); V_parts.append(v3); W_parts.append(w3)

    # A1 (row 0) x B2 (cols 4-7), second half
    u4, v4, w4 = _embed_subblock(0, 2, 4, 1, 2, 4, u_124, v_124, w_124, n, m, p)
    U_parts.append(u4); V_parts.append(v4); W_parts.append(w4)

    # A2 (rows 1-2) x B1 (cols 0-3), first 2x2 block
    u5, v5, w5 = _embed_subblock(1, 0, 0, 2, 2, 4, u_224, v_224, w_224, n, m, p)
    U_parts.append(u5); V_parts.append(v5); W_parts.append(w5)

    # A2 (rows 1-2) x B1 (cols 0-3), second 2x2 block
    u6, v6, w6 = _embed_subblock(1, 2, 0, 2, 2, 4, u_224, v_224, w_224, n, m, p)
    U_parts.append(u6); V_parts.append(v6); W_parts.append(w6)

    # A2 (rows 1-2) x B2 (cols 4-7), first 2x2 block
    u7, v7, w7 = _embed_subblock(1, 0, 4, 2, 2, 4, u_224, v_224, w_224, n, m, p)
    U_parts.append(u7); V_parts.append(v7); W_parts.append(w7)

    # A2 (rows 1-2) x B2 (cols 4-7), second 2x2 block
    u8, v8, w8 = _embed_subblock(1, 2, 4, 2, 2, 4, u_224, v_224, w_224, n, m, p)
    U_parts.append(u8); V_parts.append(v8); W_parts.append(w8)

    U_arr = np.hstack(U_parts)
    V_arr = np.hstack(V_parts)
    W_arr = np.hstack(W_parts)

    return U_arr, V_arr, W_arr


def _construct_schoolbook(n=3, m=4, p=8):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Fallback: Schoolbook (rank 96)
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p

    # Try Rank 88 hybrid decomposition
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass

    return best_result
