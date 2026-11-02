"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_strassen_222():
    """Return U,V,W (each 4x7) for 2x2x2 matrix multiplication (Strassen rank 7)."""
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
    ], dtype=float)
    W = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    return U, V, W


def _build_strassen_223():
    """Return U,V,W for 2x2x3 multiplication (Strassen 7 + 2 naive = rank 9)."""
    U222, V222, W222 = _build_strassen_222()
    # naive terms for the 3rd column of B
    Un = np.zeros((4, 2)); Vn = np.zeros((4, 2)); Wn = np.zeros((4, 2))
    Un[0, 0] = 1.0; Vn[2, 0] = 1.0; Wn[0, 0] = 1.0   # (0,1,2)
    Un[2, 1] = 1.0; Vn[3, 1] = 1.0; Wn[2, 1] = 1.0   # (1,1,2)
    return np.hstack([U222, Un]), np.hstack([V222, Vn]), np.hstack([W222, Wn])


def _build_block_decomposition(n, m, p):
    """Partition into 2x2 blocks; use Strassen(7) for full 2x2x2, Strassen+naive(9) for 2x2x3."""
    UB, VB, WB = [], [], []
    for ib in range(n // 2):
        for jb in range(m // 2):
            for kb in range(p // 2):
                U, V, W = _build_strassen_222()
                UB.append(U); VB.append(V); WB.append(W)
            for kb in range(p // 2, p // 2 + (p % 2)):
                U, V, W = _build_strassen_223()
                UB.append(U); VB.append(V); WB.append(W)
    return np.hstack(UB), np.hstack(VB), np.hstack(WB)


def _verify(U, V, W, n, m, p, rank, tol=1e-6):
    """Return max absolute error of the decomposition vs the all-ones tensor."""
    worst = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :rank] * V[j * p + k, :rank] * W[k * n + i, :rank])
                e = abs(val - 1.0)
                if e > worst:
                    worst = e
    return worst


def _verify_fast(U, V, W, n, m, p, rank, tol=1e-6):
    """Vectorized verification. Returns max abs error."""
    worst = 0.0
    for i in range(n):
        for k in range(p):
            wi = W[k * n + i, :rank]
            for j in range(m):
                val = np.sum(U[i * m + j, :rank] * V[j * p + k, :rank] * wi)
                e = abs(val - 1.0)
                if e > worst:
                    worst = e
    return worst


def _round_half(arr):
    return np.round(arr * 2.0) / 2.0


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start = time.time()
    deadline = start + 145  # safety margin before 150s kill

    # ---- Build the structured block decomposition (rank 64) ----
    U0, V0, W0 = _build_block_decomposition(n, m, p)
    rank0 = U0.shape[1]
    err0 = _verify_fast(U0, V0, W0, n, m, p, rank0)

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    best_err = err0

    # Fallback: schoolbook rank 80 (always valid)
    def _schoolbook():
        r = n * m * p
        Us = np.zeros((n * m, r)); Vs = np.zeros((m * p, r)); Ws = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, col] = 1.0
                    Vs[j * p + k, col] = 1.0
                    Ws[k * n + i, col] = 1.0
                    col += 1
        return Us, Vs, Ws, r

    # If block decomposition somehow failed, fall back to schoolbook
    if err0 > 1e-6:
        Us, Vs, Ws, r = _schoolbook()
        best_U, best_V, best_W, best_rank, best_err = Us, Vs, Ws, r, 0.0

    # ---- Refinement: try to shave rank via structured local optimization ----
    # Strategy: from the rank-64 solution, drop a column and try to repair the
    # remaining entries by small adjustments (smooth optimization) to recover error.
    # We attempt rank 63, 62, 61, ... and accept the first that verifies cleanly.

    def _try_repair(rank, n_trials=6):
        """Attempt to produce a valid decomposition of the given rank.
        Start from best solution truncated to `rank` columns, then run a smooth
        coordinate-descent polish to drive the error to ~0. Return (U,V,W,rank,err) or None."""
        # Start: take the first `rank` columns of the current best (rank0) solution
        base_U = best_U[:, :rank].copy()
        base_V = best_V[:, :rank].copy()
        base_W = best_W[:, :rank].copy()
        base_err = _verify_fast(base_U, base_V, base_W, n, m, p, rank)
        # If already good, we are done
        if base_err < 1e-6:
            return base_U, base_V, base_W, rank, base_err

        # Smooth objective: sum of squared errors over all (i,j,k)
        # We do a lightweight alternating least-squares style update:
        # For each term r, the contribution to entry (i,j,k) is
        #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want sum_r ... = 1.
        # We optimize by cycling over r and solving for the best scalar scale of
        # each factor to reduce error (a simple but effective refinement).
        U = base_U.copy(); V = base_V.copy(); W = base_W.copy()

        def _residuals(Uu, Vv, Ww, rr):
            # vector of all (n*m*p) computed values
            vals = np.empty(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[idx] = np.sum(Uu[i * m + j, :rr] * Vv[j * p + k, :rr] * Ww[k * n + i, :rr])
                        idx += 1
            return vals

        # Precompute index maps for speed
        Uidx = np.array([i * m + j for i in range(n) for j in range(m) for _ in range(p)])
        Vidx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
        Widx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])

        def _vals_vec(Uu, Vv, Ww, rr):
            # compute all n*m*p values as a vector
            # term contribution per (i,j,k,r)
            out = np.zeros(n * m * p)
            for r in range(rr):
                ucol = Uu[:, r]
                vcol = Vv[:, r]
                wcol = Ww[:, r]
                # out[t] += ucol[Uidx[t]] * vcol[Vidx[t]] * wcol[Widx[t]]
                out += ucol[Uidx] * vcol[Vidx] * wcol[Widx]
            return out

        # Gradient-free coordinate polish: for each r, find optimal scalar multipliers
        # applied to U[:,r], V[:,r], W[:,r] to reduce ||vals-1||^2.
        # We use a small grid search over a combined scale s: U[:,r]*s, V[:,r]*s, W[:,r]*s.
        # (Scaling all three by s multiplies the term by s^3.)
        for _sweep in range(4):
            vals = _vals_vec(U, V, W, rank)
            for r in range(rank):
                # current contribution of term r
                contrib = U[:, r][Uidx] * V[:, r][Vidx] * W[:, r][Widx]
                # residual without term r
                res_wo = vals - contrib
                # we want to choose s to minimize ||res_wo + s^3 * base_contrib||^2
                # where base_contrib = U[:,r]*V[:,r]*W[:,r] (current, s=1)
                # Let c = base_contrib. Minimize sum_t (res_wo[t] + s^3 c[t])^2
                # d/ds = 2 s^2 sum_t 3 c[t] (res_wo[t] + s^3 c[t]) = 0
                # => sum_t c[t] (res_wo[t] + s^3 c[t]) = 0
                # => sum c*res + s^3 sum c^2 = 0
                # => s^3 = -sum(c*res_wo)/sum(c^2)
                s3 = -np.sum(contrib * res_wo) / (np.sum(contrib * contrib) + 1e-30)
                if s3 <= 0:
                    continue
                s = s3 ** (1.0 / 3.0)
                # limit the step to avoid blow-up
                s = max(0.5, min(2.0, s))
                U[:, r] *= s
                V[:, r] *= s
                W[:, r] *= s
            # recompute error
            cur_err = np.max(np.abs(_vals_vec(U, V, W, rank) - 1.0))
            if cur_err < 1e-6:
                break

        err = _verify_fast(U, V, W, n, m, p, rank)
        if err < 1e-6:
            return U, V, W, rank, err
        return None

    # Try to reduce rank one step at a time from the current best.
    for target_rank in range(best_rank - 1, max(best_rank - 6, 50), -1):
        if time.time() > deadline - 20:
            break
        # Only attempt if we currently have a clean solution to build from
        if best_err > 1e-6:
            break
        res = _try_repair(target_rank)
        if res is not None:
            U, V, W, r, e = res
            best_U, best_V, best_W, best_rank, best_err = U, V, W, r, e
        # else keep current best and continue to next (lower) target
        if time.time() > deadline - 20:
            break

    # ---- Final: round to nearest integer / half-integer and verify ----
    def _round_and_verify(U, V, W, rank):
        Ur = _round_half(U[:, :rank])
        Vr = _round_half(V[:, :rank])
        Wr = _round_half(W[:, :rank])
        e = _verify_fast(Ur, Vr, Wr, n, m, p, rank)
        return Ur, Vr, Wr, e

    # Round the best solution
    Ur, Vr, Wr, er = _round_and_verify(best_U, best_V, best_W, best_rank)
    if er > 1e-6:
        # Rounding broke it; use the unrounded floats (still valid numerically)
        Ur, Vr, Wr = best_U[:, :best_rank], best_V[:, :best_rank], best_W[:, :best_rank]
        er = best_err
    else:
        best_U, best_V, best_W, best_err = Ur, Vr, Wr, er

    # If rounding/repair failed entirely, fall back to schoolbook
    if best_err > 1e-6:
        Us, Vs, Ws, r = _schoolbook()
        Ur, Vr, Wr, er = _round_and_verify(Us, Vs, Ws, r)
        best_U, best_V, best_W, best_rank, best_err = Ur, Vr, Wr, r, er

    # ---- Convert to lists with integer / half-integer entries ----
    def _to_list(arr):
        out = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < 1e-6:
                    row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row.append(nearest_half)
                else:
                    row.append(nearest_half)
            out.append(row)
        return out

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
# EVOLVE-BLOCK-END
