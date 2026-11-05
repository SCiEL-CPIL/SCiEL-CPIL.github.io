import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = 118.0

    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Double Strassen definitions
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),
    ]

    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2 + c
            elif b_idx == 2: gi, gj = 2 + r, c
            else: gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2 + c
            elif b_idx == 2: gj, gk = 2 + r, c
            else: gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list, v_list, w_list = [], [], []
            for p_elem, coeff in p_elems:
                u_list.extend((gi, gj, coeff * c) for gi, gj, c in get_p_element(p_elem, p_bc))
            for q_elem, coeff in q_elems:
                v_list.extend((gj, gk, coeff * c) for gj, gk, c in get_q_element(q_elem, q_bc))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    w_list.append((bc * 2 + lc, br * 2 + lr, sign * local_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1

    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
