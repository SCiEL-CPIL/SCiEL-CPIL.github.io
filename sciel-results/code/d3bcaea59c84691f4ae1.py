import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=12):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n * m) @ Ua)
                y = (B.reshape(m * p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                # C_expected is (n, p). We need (p, n) for W indices [k*n+i]
                C_expected_T = C_expected.T
                if not np.allclose(C_computed, C_expected_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    # Build rank 65 decomposition: Strassen (49) + Schoolbook (16)
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

    # Inner Strassen 2x2x2
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),             # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),             # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),             # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),             # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),               # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),               # M7
    ]

    # Super Strassen 4x4x4
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),      # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),      # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),      # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),      # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),              # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),              # M7
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify
    ok = verify(U, V, W, seed=99, trials=8)
    if not ok:
        # Fallback: Schoolbook rank 80
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i * 4 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
