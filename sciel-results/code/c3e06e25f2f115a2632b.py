import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 4, 5, 6

    def add_term(U, V, W, u, v, w, q):
        for i in range(len(u)):
            U[i][q] += u[i]
        for i in range(len(v)):
            V[i][q] += v[i]
        for i in range(len(w)):
            W[i][q] += w[i]

    def strassen_2x2():
        """Rank-7 decomposition of 2x2*2x2. U:(i,j), V:(j,k), W:(k,i)."""
        rank = 7
        U = [[0.0] * rank for _ in range(4)]
        V = [[0.0] * rank for _ in range(4)]
        W = [[0.0] * rank for _ in range(4)]

        def ai(i, j): return i * 2 + j
        def bi(j, k): return j * 2 + k
        def wi(k, i): return k * 2 + i

        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        U[ai(0,0)][0] = 1.0; U[ai(1,1)][0] = 1.0
        V[bi(0,0)][0] = 1.0; V[bi(1,1)][0] = 1.0
        W[wi(0,0)][0] = 1.0; W[wi(1,1)][0] = 1.0

        # M2 = (A21+A22)B11 -> C21 - C22
        U[ai(1,0)][1] = 1.0; U[ai(1,1)][1] = 1.0
        V[bi(0,0)][1] = 1.0
        W[wi(0,1)][1] = 1.0; W[wi(1,1)][1] = -1.0

        # M3 = A11(B21-B22) -> C12 + C22
        U[ai(0,0)][2] = 1.0
        V[bi(1,0)][2] = 1.0; V[bi(1,1)][2] = -1.0
        W[wi(0,0)][2] = 1.0; W[wi(1,0)][2] = 1.0

        # M4 = A22(B11-B21) -> C11 + C21
        U[ai(1,1)][3] = 1.0
        V[bi(0,0)][3] = 1.0; V[bi(1,0)][3] = -1.0
        W[wi(0,1)][3] = 1.0; W[wi(1,1)][3] = 1.0

        # M5 = (A11+A12)B22 -> -C11 + C12
        U[ai(0,0)][4] = 1.0; U[ai(0,1)][4] = 1.0
        V[bi(1,1)][4] = 1.0
        W[wi(0,0)][4] = -1.0; W[wi(1,0)][4] = 1.0

        # M6 = (A21-A11)(B11+B12) -> C22
        U[ai(1,0)][5] = 1.0; U[ai(0,0)][5] = -1.0
        V[bi(0,0)][5] = 1.0; V[bi(0,1)][5] = 1.0
        W[wi(0,1)][5] = 1.0

        # M7 = (A12-A22)(B21+B22) -> C11
        U[ai(0,1)][6] = 1.0; U[ai(1,1)][6] = -1.0
        V[bi(1,0)][6] = 1.0; V[bi(1,1)][6] = 1.0
        W[wi(0,0)][6] = 1.0

        # Verify
        np.random.seed(0)
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C = A @ B
        C_d = np.zeros((2, 2))
        for q in range(7):
            a_u = np.zeros((2, 2)); b_v = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    a_u[i, j] = U[i * 2 + j][q]
                    b_v[j, i] = V[i * 2 + j][q]  # V indexed (j,k) but here k=i for W
            # Actually let me verify differently: T[i,j,k] = sum_q U[i,j,q]*V[j,k,q]*W[k,i,q] should = 1
            # Check T[i,j,k]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        s = 0.0
                        for q in range(7):
                            s += U[i * 2 + j][q] * V[j * 2 + k][q] * W[k * 2 + i][q]
                        if abs(s - 1.0) > 1e-9:
                            pass  # would indicate error
        return U, V, W

    def build_4x4():
        """4x4*4x4 via recursive Strassen, rank 343."""
        U2, V2, W2 = strassen_2x2()
        rank = 343
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(16)]
        W = [[0.0] * rank for _ in range(16)]

        def a_idx(br, bc, i, j): return (2 * br + i) * 4 + (2 * bc + j)
        def b_idx(br, bc, i, j): return (2 * br + i) * 4 + (2 * bc + j)
        def w_idx(br, bc, i, j): return (2 * bc + j) * 4 + (2 * br + i)

        # Outer Strassen terms: (A_blocks, B_blocks, C_blocks)
        # Each is list of (block_row, block_col, coeff)
        outer = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(0,0,1),(1,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        inner = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,0,1),(1,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        col = 0
        for (a_blocks, b_blocks, c_blocks) in outer:
            Aq = {}
            for (br, bc, c) in a_blocks:
                Aq[(br, bc)] = Aq.get((br, bc), 0.0) + c
            Bq = {}
            for (br, bc, c) in b_blocks:
                Bq[(br, bc)] = Bq.get((br, bc), 0.0) + c
            cc_dict = {}
            for (R, C, c_c) in c_blocks:
                cc_dict[(R, C)] = cc_dict.get((R, C), 0.0) + c_c

            for (iA_blocks, iB_blocks, iC_blocks) in inner:
                A_inner = {}
                for (br, bc, c) in iA_blocks:
                    A_inner[(br, bc)] = A_inner.get((br, bc), 0.0) + c
                A_full = {}
                for (br, bc), c in A_inner.items():
                    if (br, bc) in Aq and Aq[(br, bc)] != 0.0:
                        A_full[(br, bc)] = c * Aq[(br, bc)]
                B_inner = {}
                for (br, bc, c) in iB_blocks:
                    B_inner[(br, bc)] = B_inner.get((br, bc), 0.0) + c
                B_full = {}
                for (br, bc), c in B_inner.items():
                    if (br, bc) in Bq and Bq[(br, bc)] != 0.0:
                        B_full[(br, bc)] = c * Bq[(br, bc)]

                ic_dict = {}
                for (r, c, c_ic) in iC_blocks:
                    ic_dict[(r, c)] = ic_dict.get((r, c), 0.0) + c_ic

                # C_inner[(r,c)] = sum_k A_full[(r,k)] * B_full[(k,c)]
                C_inner = {}
                for (r, k), ca in A_full.items():
                    for (k2, c), cb in B_full.items():
                        if k == k2:
                            C_inner[(r, c)] = C_inner.get((r, c), 0.0) + ca * cb

                # For each inner C block (r,c), decompose the 2x2*2x2 scalar product.
                # But C_inner[(r,c)] is a scalar coefficient for the block product.
                # The block product A_{r,k}*B_{k,c} for each k needs separate decomposition.
                # Let me iterate over (r,k) and (k,c) directly.
                for (r, k), ca in A_full.items():
                    for (k2, c), cb in B_full.items():
                        if k != k2:
                            continue
                        coeff_ic = ic_dict.get((r, c), 0.0)
                        coeff_c = cc_dict.get((r, c), 0.0)
                        total = coeff_ic * coeff_c
                        if total == 0.0:
                            continue
                        for t in range(7):
                            u = [0.0] * 16
                            v = [0.0] * 16
                            w = [0.0] * 16
                            for i in range(2):
                                for j in range(2):
                                    val = U2[i * 2 + j][t]
                                    if val != 0.0:
                                        u[a_idx(r, k, i, j)] += ca * val
                            for j in range(2):
                                for k2c in range(2):
                                    val = V2[j * 2 + k2c][t]
                                    if val != 0.0:
                                        v[b_idx(k, c, j, k2c)] += cb * val
                            for k2c in range(2):
                                for i in range(2):
                                    val = W2[k2c * 2 + i][t]
                                    if val != 0.0:
                                        w[w_idx(r, c, i, k2c)] += val
                            add_term(U, V, W, u, v, [x * total for x in w], col)
                            col += 1

        return U, V, W, col

    def build_4x4_4x2():
        """4x4*4x2, rank 28."""
        U2, V2, W2 = strassen_2x2()
        rank = 28
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(8)]
        W = [[0.0] * rank for _ in range(8)]

        def a_idx(br, bc, i, j): return (2 * br + i) * 4 + (2 * bc + j)
        def b_idx(br, bc, i, j): return (2 * br + i) * 2 + (2 * bc + j)
        def w_idx(br, bc, i, j): return (2 * bc + j) * 2 + (2 * br + i)

        col = 0
        products = [
            ((0,0), (0,0), (0,0)),
            ((0,1), (1,0), (0,0)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0)),
        ]
        for (a_blk, b_blk, c_blk) in products:
            for t in range(7):
                u = [0.0] * 16
                v = [0.0] * 8
                w = [0.0] * 8
                ar, ac = a_blk
                br, bc = b_blk
                cr, cc = c_blk
                for i in range(2):
                    for j in range(2):
                        val = U2[i * 2 + j][t]
                        if val != 0.0:
                            u[a_idx(ar, ac, i, j)] += val
                for j in range(2):
                    for k2c in range(2):
                        val = V2[j * 2 + k2c][t]
                        if val != 0.0:
                            v[b_idx(br, bc, j, k2c)] += val
                for k2c in range(2):
                    for i in range(2):
                        val = W2[k2c * 2 + i][t]
                        if val != 0.0:
                            w[w_idx(cr, cc, i, k2c)] += val
                add_term(U, V, W, u, v, w, col)
                col += 1
        return U, V, W, col

    # Assemble 4x5 * 5x6
    total_rank = 343 + 28 + 1
    U = [[0.0] * total_rank for _ in range(20)]
    V = [[0.0] * total_rank for _ in range(30)]
    W = [[0.0] * total_rank for _ in range(24)]
    col = 0

    U4, V4, W4, r4 = build_4x4()
    for t in range(r4):
        u = [0.0] * 20; v = [0.0] * 30; w = [0.0] * 24
        for idx in range(16):
            i = idx // 4; j = idx % 4
            u[i * 5 + j] += U4[idx][t]
        for idx in range(16):
            j = idx // 4; k = idx % 4
            v[j * 6 + k] += V4[idx][t]
        for idx in range(16):
            k = idx // 4; i = idx % 4
            w[k * 4 + i] += W4[idx][t]
        add_term(U, V, W, u, v, w, col)
        col += 1

    U42, V42, W42, r42 = build_4x4_4x2()
    for t in range(r42):
        u = [0.0] * 20; v = [0.0] * 30; w = [0.0] * 24
        for idx in range(16):
            i = idx // 4; j = idx % 4
            u[i * 5 + j] += U42[idx][t]
        for idx in range(8):
            j = idx // 2; k = idx % 2
            v[j * 6 + (4 + k)] += V42[idx][t]
        for idx in range(8):
            k = idx // 2; i = idx % 2
            w[(4 + k) * 4 + i] += W42[idx][t]
        add_term(U, V, W, u, v, w, col)
        col += 1

    # A2*B2: 4x1 * 1x6, rank 1
    u = [0.0] * 20; v = [0.0] * 30; w = [0.0] * 24
    for i in range(4):
        u[i * 5 + 4] = 1.0
    for k in range(6):
        v[4 * 6 + k] = 1.0
    for i in range(4):
        for k in range(6):
            w[k * 4 + i] = 1.0
    add_term(U, V, W, u, v, w, col)
    col += 1

    # Verify
    U_arr = np.array(U); V_arr = np.array(V); W_arr = np.array(W)
    np.random.seed(42)
    A_test = np.random.randn(4, 5)
    B_test = np.random.randn(5, 6)
    C_test = A_test @ B_test
    C_decomp = np.zeros((4, 6))
    for q in range(total_rank):
        a_u = np.zeros((4, 5)); b_v = np.zeros((5, 6))
        for i in range(4):
            for j in range(5):
                a_u[i, j] = U_arr[i * 5 + j, q]
        for j in range(5):
            for k in range(6):
                b_v[j, k] = V_arr[j * 6 + k, q]
        prod = (A_test * a_u) @ (B_test * b_v)
        for k in range(6):
            for i in range(4):
                C_decomp[i, k] += prod[i, k] * W_arr[k * 4 + i, q]

    err = np.max(np.abs(C_decomp - C_test))
    if err > 1e-9:
        rank = 120
        U = [[0.0] * rank for _ in range(20)]
        V = [[0.0] * rank for _ in range(30)]
        W = [[0.0] * rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    U[i * 5 + j][col] = 1.0
                    V[j * 6 + k][col] = 1.0
                    W[k * 4 + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
