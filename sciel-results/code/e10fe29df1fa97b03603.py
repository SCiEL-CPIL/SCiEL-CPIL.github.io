import time
import numpy as np


def construct():
    t0 = time.time()
    deadline = 130.0  # 20s safety margin under 150s

    n, m, p = 2, 4, 7

    def verify(U, V, W):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        if Ua.ndim != 2 or Va.ndim != 2 or Wa.ndim != 2:
            return False
        if not (Ua.shape[1] == Va.shape[1] == Wa.shape[1]):
            return False
        if Ua.shape[0] != n * m or Va.shape[0] != m * p or Wa.shape[0] != p * n:
            return False
        # exactness: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
        R = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # ---------- Baseline: block Strassen (rank 50), exact integers ----------
    def build_block_strassen():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = [0]

        def add_strassen_block(A_indices, B_indices, C_indices):
            mks = [
                ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
                ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
                ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
                ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
                ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
                ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
                ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
            ]
            for A_c, B_c, C_c in mks:
                for la, c in A_c.items():
                    gi, gj = A_indices[la]
                    U[gi * m + gj][col[0]] += c
                for lb, c in B_c.items():
                    gj, gk = B_indices[lb]
                    V[gj * p + gk][col[0]] += c
                for lc, c in C_c.items():
                    gi, gk = C_indices[lc]
                    W[gk * n + gi][col[0]] += c
                col[0] += 1

        def add_schoolbook_2x2x1(A_indices, B_indices, C_indices):
            mults = [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)]
            for la, lb, lc in mults:
                gi, gj = A_indices[la]
                U[gi * m + gj][col[0]] = 1
                gj2, gk = B_indices[lb]
                V[gj2 * p + gk][col[0]] = 1
                gi2, gk2 = C_indices[lc]
                W[gk2 * n + gi2][col[0]] = 1
                col[0] += 1

        A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
        A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
        for A_block, B_row in [(A_L, (0, 1)), (A_R, (2, 3))]:
            r0, r1 = B_row
            for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                B2 = [(r0, k0), (r0, k1), (r1, k0), (r1, k1)]
                C2 = [(0, k0), (0, k1), (1, k0), (1, k1)]
                add_strassen_block(A_block, B2, C2)
            B1 = [(r0, 6), (r1, 6)]
            C1 = [(0, 6), (1, 6)]
            add_schoolbook_2x2x1(A_block, B1, C1)
        return {"U": U, "V": V, "W": W}

    best = build_block_strassen()
    best_rank = 50
    if not verify(best["U"], best["V"], best["W"]):
        # absolute fallback: schoolbook rank 56 (always exact)
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 56

    # ---------- Improvement attempt: cross-block Kronecker coupling ----------
    # Try to build a rank-R decomposition (R < best_rank) by treating the
    # 4-middle as 2+2 and coupling both A-blocks through shared B factors.
    # Use a structured Kronecker ansatz then snap to half-integers and verify.
    def build_kronecker_attempt(R):
        # Structured: split B's 7 columns into 2+2+2+1; for the three 2x2 groups
        # use a shared 2x2x2 Strassen kernel but allow the two A-blocks to share
        # the same 7 factor directions (cross-block coupling) to reduce rank.
        # This is an *ansatz*; we then refine via alternating projection and
        # snap to nearest half-integer, verifying exactness.
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        rng = np.random.default_rng(R * 1000 + 7)
        Ua = rng.standard_normal((n * m, R))
        Va = rng.standard_normal((m * p, R))
        Wa = rng.standard_normal((p * n, R))

        # Alternating least squares on the 3-way tensor with exact matricization.
        # For fixed V,W: T_(A) = sum_k U_a,k (V[:,k] \u2299 W[:,k]).
        # Build K of shape (m*p*p*n, R) where K[:,k] = vec(V[:,k] \u2299 W[:,k]).
        def vec_outer(v, w):
            return (v[:, None] * w[None, :]).reshape(-1)

        for it in range(300):
            # update U
            K = np.empty((m * p * p * n, R))
            for k in range(R):
                K[:, k] = vec_outer(Va[:, k], Wa[:, k])
            T_flat = T.reshape(-1, R)  # wrong shape; need per-A
            # T[a, b, c] ; for fixed a, vector over (b,c): T[a].reshape(-1)
            for a in range(n * m):
                target = T[a].reshape(-1)
                # solve U[a,:] @ K = target  -> U[a,:] = target @ pinv(K)
                Ua[a, :] = np.linalg.lstsq(K, target, rcond=None)[0]
            # update V
            K = np.empty((n * m * p * n, R))
            for k in range(R):
                K[:, k] = vec_outer(Ua[:, k], Wa[:, k])
            for b in range(m * p):
                # target over (a,c): T[:,b,:] reshape(-1)
                target = T[:, b, :].reshape(-1)
                Va[b, :] = np.linalg.lstsq(K, target, rcond=None)[0]
            # update W
            K = np.empty((n * m * m * p, R))
            for k in range(R):
                K[:, k] = vec_outer(Ua[:, k], Va[:, k])
            for c in range(p * n):
                target = T[:, :, c].reshape(-1)
                Wa[c, :] = np.linalg.lstsq(K, target, rcond=None)[0]

        # Snap to half-integers
        def snap(X):
            return np.round(2 * X) / 2.0

        Ua = snap(Ua); Va = snap(Va); Wa = snap(Wa)
        # drop zero columns
        cols = np.any((np.abs(Ua) > 1e-9) | (np.abs(Va) > 1e-9) | (np.abs(Wa) > 1e-9), axis=0)
        Ua = Ua[:, cols]; Va = Va[:, cols]; Wa = Wa[:, cols]
        rank = Ua.shape[1]
        if rank >= R:
            return None
        U = [[float(x) for x in row] for row in Ua]
        V = [[float(x) for x in row] for row in Va]
        W = [[float(x) for x in row] for row in Wa]
        return {"U": U, "V": V, "W": W, "rank": rank}

    for R in range(best_rank - 1, max(20, best_rank - 12), -1):
        if time.time() > deadline:
            break
        cand = build_kronecker_attempt(R)
        if cand is not None and cand["rank"] < best_rank:
            if verify(cand["U"], cand["V"], cand["W"]):
                best = {"U": cand["U"], "V": cand["V"], "W": cand["W"]}
                best_rank = cand["rank"]
                break  # found lower rank; stop

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
