"""Tensor decomposition for 4x4 * 4x7 matrix multiplication.
Combined approach: structured Strassen block decomposition (rank 76) + coordinate-descent
refinement to reduce rank below 76, with half-integer rounding."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def _elapsed_ok():
        return time.time() < deadline

    def make_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_2x2x2():
        """Strassen's 7-product decomposition for 2x2 * 2x2 = 2x2.
        Returns U (4x7), V (4x7), W (4x7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def _map2x2_to4x4p(p_val, U, V, W, off, r):
        """Place a 2x2 * 2xp decomposition (U:(4,r), V:(2p,r), W:(2p,r)) into a 4x4 * 4xp
        tensor at column offset off. The 2x2 A-block maps to A indices [0,1,4,5];
        the 2xp B-block (rows j=0,1) maps to B indices j*4+k; the 2xp C-block (rows i=0,1)
        maps to C indices k*4+i."""
        a_map = [0, 1, 4, 5]
        for idx_a, idx in enumerate(a_map):
            U[idx, off:off+r] = U[idx_a, :]
        for j in range(2):
            for k in range(p_val):
                V[j*4 + k, off:off+r] = V[j*p_val + k, :]
        for k in range(p_val):
            for i in range(2):
                W[k*4 + i, off:off+r] = W[k*p_val + i, :]

    def strassen_4x4x4():
        """4x4 * 4x4 = 4x4 via 4 independent Strassen 2x2x2 blocks. Rank 4*7 = 28.
        Returns U (16x28), V (16x28), W (16x28)."""
        U = np.zeros((16, 28))
        V = np.zeros((16, 28))
        W = np.zeros((16, 28))
        off = 0
        for jblk in range(2):
            for iblk in range(2):
                Us, Vs, Ws = strassen_2x2x2()
                _map2x2_to4x4p(2, U, V, W, off, 7)
                off += 7
        return U, V, W

    def schoolbook_4x4x3():
        """4x4 * 4x3 = 4x3 schoolbook. Rank 4*4*3 = 48.
        Returns U (16x48), V (12x48), W (12x48)."""
        U = np.zeros((16, 48))
        V = np.zeros((12, 48))
        W = np.zeros((12, 48))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4 + j, col] = 1
                    V[j*3 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
        return U, V, W

    def decompose_4x4x7():
        """Block decomposition: 4x4x4 Strassen (rank 28) + 4x4x3 schoolbook (rank 48). Total 76.
        U (16x76), V (28x76), W (28x76)."""
        r1 = 28
        r2 = 48
        total = r1 + r2
        U = np.zeros((16, total))
        V = np.zeros((28, total))
        W = np.zeros((28, total))

        U1, V1, W1 = strassen_4x4x4()
        U[:, :r1] = U1
        V[:16, :r1] = V1
        W[:16, :r1] = W1

        U2, V2, W2 = schoolbook_4x4x3()
        U[:, r1:] = U2
        V[16:, r1:] = V2
        W[16:, r1:] = W2
        return U, V, W, total

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
        result = prod.sum(axis=1)
        return float(np.max(np.abs(result - 1.0)))

    def round_to_half(U, V, W):
        """Round entries to nearest half-integer; keep if still valid (small error)."""
        Ur = np.round(2 * np.array(U, dtype=float)) / 2.0
        Vr = np.round(2 * np.array(V, dtype=float)) / 2.0
        Wr = np.round(2 * np.array(W, dtype=float)) / 2.0
        if verify(Ur, Vr, Wr) < 1e-7:
            return Ur, Vr, Wr
        return None

    def try_reduce_rank(U, V, W, current_rank, target_rank, max_iter=5000, tol=1e-8):
        """Attempt to reduce rank by zeroing out columns via coordinate descent.
        Start from the given decomposition and iteratively try to drive specific
        columns toward zero while maintaining validity. Returns (U,V,W, new_rank) or None."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()

        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx

        def cur_err():
            prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
            res = prod.sum(axis=1)
            return float(np.max(np.abs(res - 1.0)))

        def update_row_U(a):
            ts = np.where(u_idx == a)[0]
            if len(ts) == 0:
                return
            cur = (Ua[a, :] * Va[v_idx[ts], :] * Wa[w_idx[ts], :]).sum(axis=1)
            e = cur - 1.0
            vw = Va[v_idx[ts], :] * Wa[w_idx[ts], :]
            num = (e[:, None] * vw).sum(axis=0)
            den = (vw * vw).sum(axis=0)
            mask = den > 1e-12
            Ua[a, mask] = num[mask] / den[mask]

        def update_row_V(b):
            ts = np.where(v_idx == b)[0]
            if len(ts) == 0:
                return
            cur = (Ua[u_idx[ts], :] * Va[b, :] * Wa[w_idx[ts], :]).sum(axis=1)
            e = cur - 1.0
            uw = Ua[u_idx[ts], :] * Wa[w_idx[ts], :]
            num = (e[:, None] * uw).sum(axis=0)
            den = (uw * uw).sum(axis=0)
            mask = den > 1e-12
            Va[b, mask] = num[mask] / den[mask]

        def update_row_W(c):
            ts = np.where(w_idx == c)[0]
            if len(ts) == 0:
                return
            cur = (Ua[u_idx[ts], :] * Va[v_idx[ts], :] * Wa[c, :]).sum(axis=1)
            e = cur - 1.0
            uv = Ua[u_idx[ts], :] * Va[v_idx[ts], :]
            num = (e[:, None] * uv).sum(axis=0)
            den = (uv * uv).sum(axis=0)
            mask = den > 1e-12
            Wa[c, mask] = num[mask] / den[mask]

        # First, do a general refinement pass
        err = cur_err()
        for it in range(200):
            if not _elapsed_ok():
                break
            for a in range(n * m):
                update_row_U(a)
            for b in range(m * p):
                update_row_V(b)
            for c in range(p * n):
                update_row_W(c)
            if (it + 1) % 50 == 0:
                err = cur_err()
                if err < tol:
                    break

        # Now try to zero out columns from the end (least important)
        new_rank = current_rank
        for target_col in range(current_rank - 1, target_rank - 1, -1):
            if not _elapsed_ok():
                break
            # Try to drive column target_col to zero
            success = False
            for trial in range(300):
                if not _elapsed_ok():
                    break
                # Zero out the target column in U, V, W
                Ua[:, target_col] = 0.0
                Va[:, target_col] = 0.0
                Wa[:, target_col] = 0.0

                # Now refine remaining columns to compensate
                for it in range(100):
                    if not _elapsed_ok():
                        break
                    for a in range(n * m):
                        update_row_U(a)
                    for b in range(m * p):
                        update_row_V(b)
                    for c in range(p * n):
                        update_row_W(c)

                err = cur_err()
                if err < 1e-6:
                    success = True
                    new_rank = target_col
                    break

            if not success:
                # Restore
                Ua[:, target_col] = 0.0
                Va[:, target_col] = 0.0
                Wa[:, target_col] = 0.0
                # Re-refine to get back to valid state
                for it in range(200):
                    if not _elapsed_ok():
                        break
                    for a in range(n * m):
                        update_row_U(a)
                    for b in range(m * p):
                        update_row_V(b)
                    for c in range(p * n):
                        update_row_W(c)
                err = cur_err()
                if err > 1e-5:
                    # Failed to recover, stop
                    break

        if new_rank < current_rank:
            Ua = Ua[:, :new_rank]
            Va = Va[:, :new_rank]
            Wa = Wa[:, :new_rank]
            if cur_err() < 1e-6:
                return Ua, Va, Wa, new_rank
        return None

    def refine(U, V, W, rank, max_iter=3000, tol=1e-9):
        """Coordinate-descent refinement over reals. Returns (U,V,W) or None if not valid."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx

        def cur_err():
            prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
            res = prod.sum(axis=1)
            return float(np.max(np.abs(res - 1.0)))

        best_err = cur_err()
        for it in range(max_iter):
            if not _elapsed_ok():
                break
            # Update U rows
            for a in range(n * m):
                ts = np.where(u_idx == a)[0]
                if len(ts) == 0:
                    continue
                cur = (Ua[a, :] * Va[v_idx[ts], :] * Wa[w_idx[ts], :]).sum(axis=1)
                e = cur - 1.0
                vw = Va[v_idx[ts], :] * Wa[w_idx[ts], :]
                num = (e[:, None] * vw).sum(axis=0)
                den = (vw * vw).sum(axis=0)
                mask = den > 1e-12
                Ua[a, mask] = num[mask] / den[mask]
            # Update V rows
            for b in range(m * p):
                ts = np.where(v_idx == b)[0]
                if len(ts) == 0:
                    continue
                cur = (Ua[u_idx[ts], :] * Va[b, :] * Wa[w_idx[ts], :]).sum(axis=1)
                e = cur - 1.0
                uw = Ua[u_idx[ts], :] * Wa[w_idx[ts], :]
                num = (e[:, None] * uw).sum(axis=0)
                den = (uw * uw).sum(axis=0)
                mask = den > 1e-12
                Va[b, mask] = num[mask] / den[mask]
            # Update W rows
            for c in range(p * n):
                ts = np.where(w_idx == c)[0]
                if len(ts) == 0:
                    continue
                cur = (Ua[u_idx[ts], :] * Va[v_idx[ts], :] * Wa[c, :]).sum(axis=1)
                e = cur - 1.0
                uv = Ua[u_idx[ts], :] * Va[v_idx[ts], :]
                num = (e[:, None] * uv).sum(axis=0)
                den = (uv * uv).sum(axis=0)
                mask = den > 1e-12
                Wa[c, mask] = num[mask] / den[mask]

            if (it + 1) % 100 == 0:
                e2 = cur_err()
                if e2 < best_err:
                    best_err = e2
                if best_err < tol:
                    break
        if best_err < 1e-6:
            return Ua, Va, Wa
        return None

    # ---- Main: build the correct block decomposition, verify, attempt refinement ----
    best_result = make_schoolbook()
    best_rank = n * m * p

    try:
        U, V, W, rank = decompose_4x4x7()
        err = verify(U, V, W)
        if err < 1e-9:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank

            # Try coordinate descent refinement
            if _elapsed_ok() and rank >= 56:
                refined = refine(U, V, W, rank, max_iter=2000)
                if refined is not None:
                    rU, rV, rW = refined
                    hr = round_to_half(rU, rV, rW)
                    if hr is not None:
                        best_result = {"U": hr[0].tolist(), "V": hr[1].tolist(), "W": hr[2].tolist()}

            # Try rank reduction
            if _elapsed_ok() and best_rank > 60:
                # Re-verify current best
                curU = np.array(best_result["U"], dtype=float)
                curV = np.array(best_result["V"], dtype=float)
                curW = np.array(best_result["W"], dtype=float)
                cur_rank = curU.shape[1]
                reduced = try_reduce_rank(curU, curV, curW, cur_rank, max(50, cur_rank - 10))
                if reduced is not None:
                    rU, rV, rW, new_rank = reduced
                    hr = round_to_half(rU, rV, rW)
                    if hr is not None:
                        best_result = {"U": hr[0].tolist(), "V": hr[1].tolist(), "W": hr[2].tolist()}
                        best_rank = new_rank
    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
