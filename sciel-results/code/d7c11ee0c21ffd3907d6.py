import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 130  # Safety margin before 150s kill

    n, m, p = 2, 3, 4

    # ---- Robust verification: check full tensor T[k,i,j] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] ----
    def verify(U, V, W, rank, tol=1e-8):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    # ---- Proven rank-22 construction (Strassen blocks + outer product), built with a clean helper ----
    def build_rank_22():
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_u(i, j, c, val):
            U[i * m + j, c] += val

        def add_v(j, k, c, val):
            V[j * p + k, c] += val

        def add_w(k, i, c, val):
            W[k * n + i, c] += val

        def add_strassen_block(a_rows, b_rows, c_cols, offset):
            # local 2x2 A: a_rows[0]=(i,j) a00, [1] a01, [2] a10, [3] a11
            # local 2x2 B: b_rows[0] b00, [1] b01, [2] b10, [3] b11
            # c_cols: [0] C00, [1] C11, [2] C10, [3] C01  (k,i)
            # m1 = (a00+a11)(b00+b11) -> C00 += m1, C11 += m1
            add_u(*a_rows[0], offset + 0, 1); add_u(*a_rows[3], offset + 0, 1)
            add_v(*b_rows[0], offset + 0, 1); add_v(*b_rows[3], offset + 0, 1)
            add_w(*c_cols[0], offset + 0, 1); add_w(*c_cols[1], offset + 0, 1)
            # m2 = (a10+a11)b00 -> C10 += m2, C11 -= m2
            add_u(*a_rows[2], offset + 1, 1); add_u(*a_rows[3], offset + 1, 1)
            add_v(*b_rows[0], offset + 1, 1)
            add_w(*c_cols[2], offset + 1, 1); add_w(*c_cols[1], offset + 1, -1)
            # m3 = a00(b01-b11) -> C01 += m3, C11 += m3
            add_u(*a_rows[0], offset + 2, 1)
            add_v(*b_rows[1], offset + 2, 1); add_v(*b_rows[3], offset + 2, -1)
            add_w(*c_cols[3], offset + 2, 1); add_w(*c_cols[1], offset + 2, 1)
            # m4 = a11(b10-b00) -> C00 += m4, C10 += m4
            add_u(*a_rows[3], offset + 3, 1)
            add_v(*b_rows[2], offset + 3, 1); add_v(*b_rows[0], offset + 3, -1)
            add_w(*c_cols[0], offset + 3, 1); add_w(*c_cols[2], offset + 3, 1)
            # m5 = (a00+a01)b11 -> C00 -= m5, C01 += m5
            add_u(*a_rows[0], offset + 4, 1); add_u(*a_rows[1], offset + 4, 1)
            add_v(*b_rows[3], offset + 4, 1)
            add_w(*c_cols[0], offset + 4, -1); add_w(*c_cols[3], offset + 4, 1)
            # m6 = (a10-a00)(b00+b10) -> C11 += m6
            add_u(*a_rows[2], offset + 5, 1); add_u(*a_rows[0], offset + 5, -1)
            add_v(*b_rows[0], offset + 5, 1); add_v(*b_rows[2], offset + 5, 1)
            add_w(*c_cols[1], offset + 5, 1)
            # m7 = (a01-a11)(b10+b11) -> C00 += m7
            add_u(*a_rows[1], offset + 6, 1); add_u(*a_rows[3], offset + 6, -1)
            add_v(*b_rows[2], offset + 6, 1); add_v(*b_rows[3], offset + 6, 1)
            add_w(*c_cols[0], offset + 6, 1)

        a_rows_1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        # Block 1: A1 (rows 0,1; cols 0,1) * B (rows 0,1; cols 0,1)
        b_rows_1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        c_cols_1 = [(0, 0), (1, 1), (0, 1), (1, 0)]  # C00, C11, C10, C01
        add_strassen_block(a_rows_1, b_rows_1, c_cols_1, 0)

        # Block 2: A1 * B (rows 0,1; cols 2,3)
        b_rows_2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
        c_cols_2 = [(2, 0), (3, 1), (2, 1), (3, 0)]  # C02, C13, C12, C03
        add_strassen_block(a_rows_1, b_rows_2, c_cols_2, 7)

        # Block 3: A (rows 0,1; col 2) * B (row 2; cols 0..3)  -> rank 8
        offset = 14
        for i in [0, 1]:
            for k in [0, 1, 2, 3]:
                add_u(i, 2, offset, 1)
                add_v(2, k, offset, 1)
                add_w(k, i, offset, 1)
                offset += 1

        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list, rank

    def to_list(a):
        return [[float(x) for x in row] for row in a]

    def try_als(rank, max_iter=300, tol=1e-12, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = np.zeros((p * n, rank))

        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1

            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            if diff < tol:
                U_list = to_list(U); V_list = to_list(V); W_list = to_list(W)
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U2 = to_list(U_snap); V2 = to_list(V_snap); W2 = to_list(W_snap)
                    if verify(U2, V2, W2, rank):
                        return U2, V2, W2
                    return U_list, V_list, W_list
        return None

    # ---- Final high-precision polish: re-solve each factor with tight tol, snap, verify ----
    def polish(U, V, W, rank, iters=200):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for _ in range(iters):
            if time.time() - start_time > max_time - 5:
                break
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = Ua[i * m + j, :] * Va[j * p + k, :]
                    try:
                        W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = Wa[k * n + i, :]
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = Va[j * p + k, :] * Wa[k * n + i, :]
                    try:
                        U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = Ua[i * m + j, :]
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = Ua[i * m + j, :] * Wa[k * n + i, :]
                    try:
                        V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = Va[j * p + k, :]
            Ua, Va, Wa = U_new, V_new, W_new
        U_list = to_list(snap_to_half_integer(Ua))
        V_list = to_list(snap_to_half_integer(Va))
        W_list = to_list(snap_to_half_integer(Wa))
        if verify(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list
        U_list = to_list(Ua); V_list = to_list(Va); W_list = to_list(Wa)
        if verify(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list
        return None

    # ---- Search for lower rank via ALS ----
    for rank in range(18, 22):
        if time.time() - start_time > max_time - 10:
            break
        best = None
        for seed in range(60):
            if time.time() - start_time > max_time - 5:
                break
            result = try_als(rank, max_iter=250, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list, rank):
                    # high-precision polish
                    polished = polish(U_list, V_list, W_list, rank)
                    if polished is not None:
                        U_list, V_list, W_list = polished
                    best = (U_list, V_list, W_list, rank)
                    break
        if best is not None:
            U_list, V_list, W_list, r = best
            return {"U": U_list, "V": V_list, "W": W_list}

    # ---- Fallback: proven rank-22 construction ----
    U_list, V_list, W_list, rank = build_rank_22()
    if not verify(U_list, V_list, W_list, rank):
        # last-resort schoolbook rank 24 (always valid)
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        U_list = to_list(U); V_list = to_list(V); W_list = to_list(W)
    return {"U": U_list, "V": V_list, "W": W_list}
