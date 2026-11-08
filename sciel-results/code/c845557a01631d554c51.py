import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS

    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = (rank_sb, U_sb, V_sb, W_sb)

    def verify(U, V, W):
        Um = np.array(U, dtype=float)
        Vm = np.array(V, dtype=float)
        Wm = np.array(W, dtype=float)
        Um = Um.reshape(n, m, -1)
        Vm = Vm.reshape(m, p, -1)
        Wm = Wm.reshape(p, n, -1)
        T = np.zeros((n, m, p))
        r = Um.shape[2]
        for c in range(r):
            uc = Um[:, :, c]
            vc = Vm[:, :, c]
            wc = Wm[:, :, c]
            wcT = wc.T
            for j in range(m):
                T[:, j, :] += uc[:, j][:, None] * vc[j, :, None] * wcT[None, :, :]
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Strassen 2x2x2
    U_strassen = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    V_strassen = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,1,1,1]
    ]
    W_strassen = [
        [1,0,0,1],
        [0,0,1,-1],
        [0,1,0,1],
        [1,0,1,0],
        [-1,1,0,0],
        [0,0,0,1],
        [1,0,0,1]
    ]

    # Block Strassen: rank 42
    rank_s = 42
    U_s = [[0] * rank_s for _ in range(8)]
    V_s = [[0] * rank_s for _ in range(24)]
    W_s = [[0] * rank_s for _ in range(12)]

    subproblems = [
        (0, 0, 0, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 0, 0, 4, 4),
        (0, 2, 2, 0, 0),
        (0, 2, 2, 2, 2),
        (0, 2, 2, 4, 4),
    ]

    for sp_idx, (a_i0, a_j0, b_j0, b_k0, c_k0) in enumerate(subproblems):
        for t in range(7):
            c = sp_idx * 7 + t
            u_vals = U_strassen[t]
            v_vals = V_strassen[t]
            w_vals = W_strassen[t]
            for li, lj in [(0,0), (0,1), (1,0), (1,1)]:
                idx = li * 4 + (a_j0 + lj)
                U_s[idx][c] = u_vals[li * 2 + lj]
            for lj, lk in [(0,0), (0,1), (1,0), (1,1)]:
                idx = (b_j0 + lj) * 6 + (b_k0 + lk)
                V_s[idx][c] = v_vals[lj * 2 + lk]
            for lk, li in [(0,0), (1,0), (0,1), (1,1)]:
                idx = (c_k0 + lk) * 2 + li
                W_s[idx][c] = w_vals[li * 2 + lk]

    if time.time() < DEADLINE and verify(U_s, V_s, W_s):
        best = (rank_s, U_s, V_s, W_s)

    def solve_for_W(U_arr, V_arr, r):
        A = np.zeros((p * n, r))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[k * n + i, c] += U_arr[i * m + j, c] * V_arr[j * p + k, c]
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A, np.ones(p * n), rcond=None)
            return W_flat.reshape(p * n, r)
        except:
            return None

    def solve_for_U(V_arr, W_arr, r):
        B = np.zeros((n * m, r))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        B[i * m + j, c] += V_arr[j * p + k, c] * W_arr[k * n + i, c]
        try:
            U_flat, _, _, _ = np.linalg.lstsq(B, np.ones(n * m), rcond=None)
            return U_flat.reshape(n * m, r)
        except:
            return None

    def solve_for_V(U_arr, W_arr, r):
        C = np.zeros((m * p, r))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C[j * p + k, c] += U_arr[i * m + j, c] * W_arr[k * n + i, c]
        try:
            V_flat, _, _, _ = np.linalg.lstsq(C, np.ones(m * p), rcond=None)
            return V_flat.reshape(m * p, r)
        except:
            return None

    def to_list(arr):
        return [[float(arr[i, c]) for c in range(arr.shape[1])] for i in range(arr.shape[0])]

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def try_rank(r, max_iters=25, n_restarts=2):
        U_s_arr = np.array(U_s, dtype=float)
        V_s_arr = np.array(V_s, dtype=float)
        W_s_arr = np.array(W_s, dtype=float)

        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15:
                return None

            if restart == 0:
                U_try = U_s_arr[:, :r].copy()
                V_try = V_s_arr[:, :r].copy()
                W_try = W_s_arr[:, :r].copy()
            else:
                np.random.seed(42 + restart)
                U_try = np.random.randn(n * m, r) * 0.5
                V_try = np.random.randn(m * p, r) * 0.5
                W_try = np.random.randn(p * n, r) * 0.5

            for it in range(max_iters):
                if time.time() > DEADLINE - 15:
                    return None

                W_new = solve_for_W(U_try, V_try, r)
                if W_new is None:
                    break
                W_try = W_new

                U_new = solve_for_U(V_try, W_try, r)
                if U_new is None:
                    break
                U_try = U_new

                V_new = solve_for_V(U_try, W_try, r)
                if V_new is None:
                    break
                V_try = V_new

            U_rounded = round_to_half(U_try)
            V_rounded = round_to_half(V_try)
            W_rounded = round_to_half(W_try)

            if verify(to_list(U_rounded), to_list(V_rounded), to_list(W_rounded)):
                return (r, to_list(U_rounded), to_list(V_rounded), to_list(W_rounded))

            if verify(to_list(U_try), to_list(V_try), to_list(W_try)):
                return (r, to_list(U_try), to_list(V_try), to_list(W_try))

        return None

    for target_rank in range(41, 23, -1):
        if time.time() > DEADLINE - 20:
            break
        result = try_rank(target_rank, max_iters=25, n_restarts=2)
        if result is not None:
            best = result
            break

    rank_best, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
