"""Matrix multiplication 4,4,5 - rank-65 decomposition (Strassen 49 + schoolbook 16)."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    t0 = time.time()
    DEADLINE = 148

    def get_schoolbook():
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() - t0 > DEADLINE:
        return get_schoolbook()

    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen for 4x4 * 4x4
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]

    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]

    inner_contribs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    super_contribs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        s_contribs = super_contribs[s_idx]

        for t_idx in range(7):
            if time.time() - t0 > DEADLINE:
                return get_schoolbook()

            p_elems, q_elems = inner_defs[t_idx]
            t_contribs = inner_contribs[t_idx]

            for p_idx, p_coeff in p_elems:
                pr = p_idx // 2
                pc = p_idx % 2
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0:
                        continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gi = base_r + pr
                    gj = base_c + pc
                    U[gi * m + gj][col] += b_coeff * p_coeff

            for q_idx, q_coeff in q_elems:
                qr = q_idx // 2
                qc = q_idx % 2
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0:
                        continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gj = base_r + qr
                    gk = base_c + qc
                    V[gj * p + gk][col] += b_coeff * q_coeff

            for s_pos in range(4):
                t_c = t_contribs[s_pos]
                if t_c == 0:
                    continue
                sr = s_pos // 2
                sc = s_pos % 2
                for c_block in range(4):
                    s_c = s_contribs[c_block]
                    if s_c == 0:
                        continue
                    total_coeff = t_c * s_c
                    if total_coeff == 0:
                        continue
                    base_i = (c_block // 2) * 2
                    base_k = (c_block % 2) * 2
                    gi = base_i + sr
                    gk = base_k + sc
                    W[gk * n + gi][col] += total_coeff

            col += 1

    # Part 2: Schoolbook for column 4 (k=4)
    for i in range(n):
        for j in range(m):
            if time.time() - t0 > DEADLINE:
                return get_schoolbook()
            U[i * m + j][col] += 1
            V[j * p + 4][col] += 1
            W[4 * n + i][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
