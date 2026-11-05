import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin before the 150s kill

    n, m, p = 2, 3, 4

    # ------------------------------------------------------------------
    # Proven rank-22 construction (Strassen on two 2x2x2 blocks + outer
    # product for the remaining column of A / row of B). This is our
    # guaranteed-valid fallback.
    # ------------------------------------------------------------------
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2,0:2] * B[0:2,2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer product: A[0:2,2:3] * B[2:3,0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    # ------------------------------------------------------------------
    # Verification (exact, integer/half-integer friendly).
    # ------------------------------------------------------------------
    def verify(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr)
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

    # ------------------------------------------------------------------
    # Smooth surrogate objective: epigraph-style soft-min over residuals.
    #   residual_t = 1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We want all residuals ~ 0. A smooth max|residual| via log-sum-exp
    # gives a differentiable penalty we can use to score near-solutions
    # and to detect convergence even before exact verify passes.
    # ------------------------------------------------------------------
    def residuals(U, V, W, rank):
        # returns array of shape (n, m, p) of (1 - product sum)
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        res = np.empty((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = 1.0 - float(np.dot(U[i * m + j, :rank],
                                                       V[j * p + k, :rank] * W[k * n + i, :rank]))
        return res

    def smooth_penalty(res, scale=50.0):
        """Smooth max |residual| using log-sum-exp (epigraph surrogate)."""
        a = np.abs(res)
        m = a.max()
        if m < 1e-12:
            return 0.0
        # soft-max of |res| scaled: (1/scale)*logsumexp(scale*a)
        sa = scale * a
        sa_max = sa.max()
        lse = sa_max + np.log(np.sum(np.exp(sa - sa_max)))
        return float(lse / scale)

    def max_residual(res):
        return float(np.max(np.abs(res))) if res.size else 0.0

    # ------------------------------------------------------------------
    # ALS: alternating least squares with a smooth-penalty-driven
    # early stop and a final high-precision polish.
    # ------------------------------------------------------------------
    def _als_solve(U, V, W, rank, max_iter, tol, seed, deadline):
        rng = np.random.RandomState(seed)
        U = np.asarray(U, dtype=float).copy()
        V = np.asarray(V, dtype=float).copy()
        W = np.asarray(W, dtype=float).copy()

        best_pen = np.inf
        best = (U.copy(), V.copy(), W.copy())
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Solve W (fix U,V)
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    b = np.ones(m)
                    try:
                        W_new[k * n + i, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = rng.randn(rank) * 0.1
            # Solve U (fix V,W)
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    b = np.ones(p)
                    try:
                        U_new[i * m + j, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = rng.randn(rank) * 0.1
            # Solve V (fix U,W)
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    b = np.ones(n)
                    try:
                        V_new[j * p + k, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = rng.randn(rank) * 0.1

            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new

            pen = smooth_penalty(residuals(U, V, W, rank))
            if pen < best_pen:
                best_pen = pen
                best = (U.copy(), V.copy(), W.copy())
            if diff < tol:
                break
        return best, best_pen

    def polish(U, V, W, rank, max_iter=400, tol=1e-15, deadline=None):
        """High-precision polish: tighter tol, more iterations, then snap."""
        U = np.asarray(U, dtype=float).copy()
        V = np.asarray(V, dtype=float).copy()
        W = np.asarray(W, dtype=float).copy()
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    try:
                        W_new[k * n + i, :], *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = W[k * n + i, :]
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    try:
                        U_new[i * m + j, :], *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = U[i * m + j, :]
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    try:
                        V_new[j * p + k, :], *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = V[j * p + k, :]
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            if diff < tol:
                break

        def to_lists(arr):
            return [[round(float(x), 12) for x in row] for row in arr]

        # Try snapped (half-integer) version first; fall back to raw.
        U_s = snap_to_half_integer(U); V_s = snap_to_half_integer(V); W_s = snap_to_half_integer(W)
        U_l, V_l, W_l = to_lists(U_s), to_lists(V_s), to_lists(W_s)
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        U_l, V_l, W_l = to_lists(U), to_lists(V), to_lists(W)
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        return None

    def try_als(rank, max_iter=250, tol=1e-12, seed=42, init=None, deadline=None):
        if init is not None:
            U0, V0, W0 = init
        else:
            rng = np.random.RandomState(seed)
            U0 = rng.randn(n * m, rank) * 0.5
            V0 = rng.randn(m * p, rank) * 0.5
            W0 = rng.randn(p * n, rank) * 0.5
        (U, V, W), pen = _als_solve(U0, V0, W0, rank, max_iter, tol, seed, deadline)
        # Accept if exact, or if smooth penalty is tiny (near-exact) -> polish.
        U_l = [[round(float(x), 10) for x in row] for row in U]
        V_l = [[round(float(x), 10) for x in row] for row in V]
        W_l = [[round(float(x), 10) for x in row] for row in W]
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        if pen < 1e-6:
            polished = polish(U, V, W, rank, deadline=deadline)
            if polished is not None:
                return polished
        return None

    def warm_start_from_rank22(target_rank, seed):
        rng = np.random.RandomState(seed)
        U22, V22, W22, _ = build_rank_22()
        Ua = np.array(U22, dtype=float); Va = np.array(V22, dtype=float); Wa = np.array(W22, dtype=float)
        cols = rng.choice(22, size=target_rank, replace=False)
        Ui = Ua[:, cols].copy(); Vi = Va[:, cols].copy(); Wi = Wa[:, cols].copy()
        Ui += rng.randn(*Ui.shape) * 0.01
        Vi += rng.randn(*Vi.shape) * 0.01
        Wi += rng.randn(*Wi.shape) * 0.01
        return Ui, Vi, Wi

    # ------------------------------------------------------------------
    # Search: try to beat rank 22. Keep best found; always have fallback.
    # ------------------------------------------------------------------
    best = None
    deadline_search = start_time + max_time - 12
    for rank in range(17, 22):
        if time.time() > deadline_search:
            break
        found = None
        # warm starts from rank-22 (structured init)
        for seed in range(30):
            if time.time() > deadline_search:
                break
            init = warm_start_from_rank22(rank, seed)
            res = try_als(rank, max_iter=200, seed=seed, init=init, deadline=deadline_search)
            if res is not None:
                U_l, V_l, W_l = res
                if verify(U_l, V_l, W_l, rank):
                    found = (U_l, V_l, W_l, rank)
                    break
        # random inits
        if found is None:
            for seed in range(30):
                if time.time() > deadline_search:
                    break
                res = try_als(rank, max_iter=200, seed=seed, deadline=deadline_search)
                if res is not None:
                    U_l, V_l, W_l = res
                    if verify(U_l, V_l, W_l, rank):
                        found = (U_l, V_l, W_l, rank)
                        break
        if found is not None:
            U_l, V_l, W_l, r = found
            polished = polish(U_l, V_l, W_l, r, deadline=start_time + max_time - 3)
            if polished is not None:
                U_l, V_l, W_l = polished
            best = (U_l, V_l, W_l, r)
            break  # lowest rank found; stop (ranks are tried ascending)

    if best is not None:
        U_l, V_l, W_l, r = best
        return {"U": U_l, "V": V_l, "W": W_l}

    # Fallback: proven rank-22 construction
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
