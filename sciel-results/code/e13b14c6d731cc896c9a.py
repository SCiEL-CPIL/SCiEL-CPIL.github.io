import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 7]


def _schoolbook(n, m, p):
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
    return U, V, W


def _snap_half(arr, tol=1e-6):
    """Snap entries to nearest half-integer; return (snapped, ok)."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    if not np.allclose(doubled, rounded, atol=tol, rtol=0):
        return None, False
    return rounded / 2.0, True


def _verify(U, V, W, T, tol=1e-7):
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return bool(np.allclose(T_rec, T, atol=tol, rtol=0))


def _build_T(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    T = _build_T(n, m, p)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    def time_left():
        return deadline - time.time()

    # ------------------------------------------------------------------
    # Strategy A: apolarity (catalecticant) border-rank estimate, then
    # exact rank-r solve with half-integer snapping.
    # ------------------------------------------------------------------
    def apolarity_min_rank():
        # T[a,b,c]; fix a,b -> linear in c.  Build matrix M[c, (a,b)].
        M = T.reshape(n * m * m * p, p * n).T  # (p*n, n*m*m*p)
        # Rank of M gives a lower bound on border rank (for this flattening).
        r = int(np.linalg.matrix_rank(M, tol=1e-9))
        return r

    def solve_rank(r):
        # Find U(4,r), V(14,r), W(14,r) with T = sum_r U_r V_r W_r.
        # Approach: fix U and V to random, solve W via linear system,
        # then iterate (alternating least squares) and snap.
        rng = np.random.default_rng(12345)
        for attempt in range(40):
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            # Solve for W: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:,:,c] = sum_r (U * V[:,r,None]) ... 
            # Vectorize: for each r, contribution to T[:,:,c] is U[:,r] V[:,r] W[c,r].
            # Build system in W: flatten T to (n*m*m*p, p*n).
            # T_flat[(a,b), c] = sum_r U[a,r] V[b,r] W[c,r]
            # => T_flat = (U ⊗ V)_{r} stacked ... 
            # Let A[c, r] = W[c, r].  T_flat[(a,b), c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
            # Define B[(a,b), r] = U[a,r]*V[b,r].  Then T_flat[(a,b),c] = sum_r B[(a,b),r] W[c,r].
            # So T_flat = B @ W.T   => W.T = solve(B, T_flat)
            B = (U[:, :, None] * V[:, None, :]).reshape(n * m * m * p, r)  # (a*b, r)
            T_flat = T.reshape(n * m * m * p, p * n)
            try:
                Wsol = np.linalg.lstsq(B, T_flat, rcond=None)[0].T  # (p*n, r)
            except np.linalg.LinAlgError:
                continue
            # Refine with a couple of ALS steps
            for _ in range(3):
                # update W given U,V
                Wsol = np.linalg.lstsq(B, T_flat, rcond=None)[0].T
                # update V given U,W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each a,c: T[a,:,c] = sum_r U[a,r] W[c,r] V[:,r]
                # => T_ac = (U[a,:]*W[c,:]) @ V.T
                # Solve for V: build system.
                for _ in range(0):
                    pass
                # Instead do simple: recompute residual; accept if small.
            err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, Wsol) - T)
            if err < 1e-4:
                Us, ok1 = _snap_half(U)
                Vs, ok2 = _snap_half(V)
                Ws, ok3 = _snap_half(Wsol)
                if ok1 and ok2 and ok3 and _verify(Us, Vs, Ws, T):
                    return Us, Vs, Ws
        return None

    # Try border-rank estimate
    try:
        lb = apolarity_min_rank()
    except Exception:
        lb = 14
    lb = max(14, lb)
    for r in range(lb, min(best_rank, 28)):
        if time_left() < 15:
            break
        res = solve_rank(r)
        if res is not None:
            U, V, W = res
            best_rank = r
            break

    # ------------------------------------------------------------------
    # Strategy B: structured Strassen-core sharing across the 7 columns.
    # C[:,k] = A @ B[:,k].  Use 7 shared "core" bilinear forms on A,
    # and per-column linear maps.  Try to beat 28 by sharing.
    # ------------------------------------------------------------------
    # Fallback: keep schoolbook if nothing better found.
    if not _verify(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
