"""Search for a rank-minimal decomposition of the 2x3x10 matrix-mult tensor.

Combines:
  * the current program's structural sparse-search + solve-for-W machinery, and
  * the inspiration's "tensor slicing" / linear-algebraic elimination idea:
    each rank term is a rank-1 slice; we exploit the per-(i,k) Khatri-Rao
    structure to systematically eliminate redundant columns and certify a
    lower-rank solution, plus a fast full-tensor verification and an exact
    rational polish pass.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 140.0  # safety margin under the 150s kill

VALS = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Fast full-tensor verification.

    Returns (ok, max_err). Checks ALL tensor entries: the n*p*m non-zero
    entries must equal 1, and every other (a,b,c) triple must equal 0.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    # Reshape to 3D tensors: Ua[i,j,r], Va[j,k,r], Wa[k,i,r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)

    # T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    target = np.ones((n, p, m))
    err_nz = float(np.max(np.abs(T - target)))

    # Zero entries: build the full (nm, mp, pn) tensor and compare to support.
    full = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    support = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support[i * m + j, j * p + k, k * n + i] = 1.0
    err_z = float(np.max(np.abs(full - support)))

    max_err = max(err_nz, err_z)
    return (max_err < tol), max_err


def _current_rank55(n, m, p):
    """Build the known rank-55 solution (seed)."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0 + 2][col] += 1; W[k1 + 2][col] -= 1
        col += 1
        U[0][col] += 1
        V[k1][col] += 1; V[10 + k1][col] -= 1
        W[k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[4][col] += 1
        V[10 + k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[0][col] += 1; U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1 + 2][col] += 1
        col += 1
        U[1][col] += 1; U[4][col] -= 1
        V[10 + k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k0][col] += 1; W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k1][col] += 1; W[k1][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}


def _solve_W(U, V, n, m, p, R):
    """Given U and V, solve for W so the tensor identity holds.

    For each (i,k): M[j,r] = U[i*m+j,r]*V[j*p+k,r]; need M @ w_ik = 1.
    Returns list-of-lists W or None if infeasible.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j] = Ua[i * m + j] * Va[j * p + k]
            try:
                w, residuals, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
            except Exception:
                return None
            if residuals.size > 0 and residuals[0] > 1e-7:
                return None
            W[k * n + i] = w
    return W.tolist()


def _rand_sparse(nrows, rank, density, rng):
    A = np.zeros((nrows, rank))
    for r_ in range(nrows):
        for c_ in range(rank):
            if rng.random() < density:
                A[r_, c_] = VALS[rng.randrange(len(VALS))]
    return A.tolist()


def _try_rank(R, n, m, p, rng, max_iters=2000, deadline=TARGET_DEADLINE):
    """Random sparse U,V + solve-for-W search for a valid rank-R solution."""
    nm, mp = n * m, m * p
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        U = _rand_sparse(nm, R, 0.20, rng)
        V = _rand_sparse(mp, R, 0.20, rng)
        W = _solve_W(U, V, n, m, p, R)
        if W is None:
            continue
        ok, _ = _verify_full(U, V, W, n, m, p)
        if ok:
            return U, V, W
    return None


def _polish_rational(U, V, W, n, m, p, tol=1e-6):
    """Snap float entries to the nearest integer/half-integer if consistent.

    Returns (U,V,W) with clean entries, or None if snapping breaks validity.
    """
    def snap(M):
        a = np.array(M, dtype=float)
        b = np.round(a * 2.0) / 2.0
        return b.tolist()

    Us, Vs, Ws = snap(U), snap(V), snap(W)
    ok, _ = _verify_full(Us, Vs, Ws, n, m, p, tol=1e-6)
    if ok:
        return Us, Vs, Ws
    return None


def _eliminate_columns(U, V, W, n, m, p, rng, deadline=TARGET_DEADLINE):
    """Tensor-slicing elimination: repeatedly try to drop one rank column.

    A column r is redundant if the tensor it contributes is a linear
    combination of the others, i.e. if we can find new U',V',W' of one
    fewer column matching the full tensor. We test this by removing column r
    and re-solving for W' (and, when needed, absorbing the residual into the
    remaining columns). A practical, exact test: drop column r, keep U,V
    unchanged (zeroed in column r), solve for W'; if the full tensor still
    matches, the column was redundant.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    R = len(U[0])
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        # Try each column; order randomized for diversity
        order = list(range(R))
        rng.shuffle(order)
        for r in order:
            if time.time() > deadline - 5:
                break
            # Candidate: delete column r
            U2 = [row[:r] + row[r + 1:] for row in U]
            V2 = [row[:r] + row[r + 1:] for row in V]
            W2 = _solve_W(U2, V2, n, m, p, R - 1)
            if W2 is None:
                continue
            ok, _ = _verify_full(U2, V2, W2, n, m, p)
            if ok:
                U, V, W = U2, V2, W2
                R -= 1
                improved = True
                break
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    # Start with the known rank-55 solution.
    best = _current_rank55(n, m, p)
    best_rank = 55
    ok, err = _verify_full(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback: trivial rank-60 solution.
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 60

    # 1) Systematically eliminate redundant columns from the seed.
    Ue, Ve, We = _eliminate_columns(best["U"], best["V"], best["W"], n, m, p, rng)
    ok, _ = _verify_full(Ue, Ve, We, n, m, p)
    if ok and len(Ue[0]) < best_rank:
        best = {"U": Ue, "V": Ve, "W": We}
        best_rank = len(Ue[0])

    # 2) Rational polish the seed (and any improved solution).
    Up, Vp, Wp = _polish_rational(best["U"], best["V"], best["W"], n, m, p)
    if Up is not None:
        best = {"U": Up, "V": Vp, "W": Wp}

    # 3) Search for lower ranks via sparse random + solve, then eliminate.
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > TARGET_DEADLINE - 15:
            break
        result = _try_rank(target_rank, n, m, p, rng, max_iters=1500)
        if result is not None:
            U, V, W = result
            U, V, W = _eliminate_columns(U, V, W, n, m, p, rng)
            ok, _ = _verify_full(U, V, W, n, m, p)
            if ok:
                Ur, Vr, Wr = _polish_rational(U, V, W, n, m, p)
                if Ur is not None:
                    U, V, W = Ur, Vr, Wr
                best = {"U": U, "V": V, "W": W}
                best_rank = len(U[0])
                break

    # Deadline guard.
    if time.time() > TARGET_DEADLINE - 3:
        return best
    return best


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify_full(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
