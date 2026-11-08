import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS

    # Rank-66 block-Strassen decomposition
    # M(3,4,6) = 3 * [M(1,4,2) + M(2,4,2)] = 3 * [8 + 14] = 66
    U_fallback = [[0] * 66 for _ in range(n * m)]
    V_fallback = [[0] * 66 for _ in range(m * p)]
    W_fallback = [[0] * 66 for _ in range(p * n)]
    current_rank = 0

    def add_rank1_fb(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= 66:
            return
        for idx, val in zip(u_indices, u_vals):
            U_fallback[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V_fallback[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W_fallback[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2_fb(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1_fb([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2_fb(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1_fb([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1_fb([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1_fb([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1_fb([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1_fb([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1_fb([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1_fb([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2_fb(i_off, j_off, k_off):
        add_2_2_2_fb(i_off, j_off, k_off)
        add_2_2_2_fb(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2_fb(0, 0, k_off)
        add_2_4_2_fb(1, 0, k_off)

    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
