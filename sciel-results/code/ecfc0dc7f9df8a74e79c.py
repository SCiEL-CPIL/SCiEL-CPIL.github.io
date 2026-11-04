"""Fresh design: search for rank-minimal exact decomposition of 4x5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
N, M, P = 4, 5, 6


def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[i,j,k] with index (i,k) in the output block."""
    T = np.zeros((n * p, m), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, j] = 1
    return T


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check sum_c U[:,c] V[:,c] reconstructs T in the W-permuted sense."""
    # U shape (n*m, r), V (m*p, r), W (p*n, r)
    # Product over c: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build T' where T'[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # More efficient: for each c, outer-like contribution
    # We'll compute directly but vectorized over c.
    max_err = 0.0
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for k in range(p):
                # sum_j uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                s = 0.0
                for j in range(m):
                    s += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
                # each c contributes to the (i,k) entry; total over c should be 1
                # accumulate into a global
                pass
    # Direct full check:
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > err:
                    err = abs(val - 1.0)
    return err < tol


def _round_half(x):
    """Round to nearest half-integer."""
    return round(2 * x) / 2.0


def _snaps_half(U, V, W):
    U2 = np.array([[_round_half(x) for x in row] for row in U], dtype=float)
    V2 = np.array([[_round_half(x) for x in row] for row in V], dtype=float)
    W2 = np.array([[_round_half(x) for x in row] for row in W], dtype=float)
    return U2, V2, W2


def _random_decomposition(rank, n, m, p, seed=0):
    """Random rank decomposition that exactly equals T, via random full-rank factors."""
    rng = np.random.default_rng(seed)
    # Build T as (n*p, m) with ones.
    T = _build_tensor(n, m, p)
    # We need U (n*m, r), V (m*p, r), W (p*n, r) such that
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Strategy: pick random U, V with full column rank, then solve for W.
    # For fixed (i,j,k), the equation is linear in W[k*n+i,c].
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    # Group by (k,i): for each (k,i), sum over j of [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]] = 1
    # This is a linear system in W. Let's set it up.
    # Variables: W[k*n+i, c] for all k,i,c. Number = p*n*r.
    # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1. Number = n*m*p.
    # Build matrix A of shape (n*m*p, p*n*r).
    A = np.zeros((n * m * p, p * n * r))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    col = (k * n + i) * r + c
                    A[eq, col] = U[i * m + j, c] * V[j * p + k, c]
                eq += 1
    # Solve least squares for W_flat
    W_flat, *_ = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape(p * n, r)
    return U, V, W


def construct():
    deadline = time.time() + 120.0
    best = None
    best_rank = N * M * P

    # Schoolbook baseline
    rank0 = N * M * P
    U = [[0] * rank0 for _ in range(N * M)]
    V = [[0] * rank0 for _ in range(M * P)]
    W = [[0] * rank0 for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    best = (U, V, W)
    best_rank = rank0

    # Try to find lower-rank decompositions via random factors + solve for W.
    # Then try to reduce rank by dropping near-zero columns / rank-revealing.
    for seed in range(200):
        if time.time() > deadline - 15.0:
            break
        for rank in range(best_rank - 1, 15, -1):
            if rank >= best_rank:
                continue
            try:
                rng = np.random.default_rng(seed)
                U0 = rng.standard_normal((N * M, rank))
                V0 = rng.standard_normal((M * P, rank))
                U0, V0, W0 = _random_decomposition(rank, N, M, P, seed=seed)
                # Actually _random_decomposition builds its own U,V; redo properly.
                U0 = rng.standard_normal((N * M, rank))
                V0 = rng.standard_normal((M * P, rank))
                # Solve for W0
                A = np.zeros((N * M * P, P * N * rank))
                b = np.ones(N * M * P)
                eq = 0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            for c in range(rank):
                                A[eq, (k * N + i) * rank + c] = U0[i * M + j, c] * V0[j * P + k, c]
                            eq += 1
                W0_flat, residuals, rankA, _ = np.linalg.lstsq(A, b, rcond=None)
                W0 = W0_flat.reshape(P * N, rank)
                # Verify
                err = 0.0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            val = 0.0
                            for c in range(rank):
                                val += U0[i * M + j, c] * V0[j * P + k, c] * W0[k * N + i, c]
                            if abs(val - 1.0) > err:
                                err = abs(val - 1.0)
                if err < 1e-6:
                    # Found a valid rank decomposition. Try to reduce rank via column dropping.
                    cur_U, cur_V, cur_W = U0.copy(), V0.copy(), W0.copy()
                    cur_rank = rank
                    improved = True
                    while improved and time.time() < deadline - 15.0:
                        improved = False
                        # Try removing each column if the residual is small
                        for c in range(cur_rank):
                            U_t = np.delete(cur_U, c, axis=1)
                            V_t = np.delete(cur_V, c, axis=1)
                            W_t = np.delete(cur_W, c, axis=1)
                            err_t = 0.0
                            for i in range(N):
                                for j in range(M):
                                    for k in range(P):
                                        val = 0.0
                                        for cc in range(cur_rank - 1):
                                            val += U_t[i * M + j, cc] * V_t[j * P + k, cc] * W_t[k * N + i, cc]
                                        if abs(val - 1.0) > err_t:
                                            err_t = abs(val - 1.0)
                            if err_t < 1e-6:
                                cur_U, cur_V, cur_W = U_t, V_t, W_t
                                cur_rank -= 1
                                improved = True
                                break
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best = (cur_U, cur_V, cur_W)
                        break
            except Exception:
                pass
        if time.time() > deadline - 15.0:
            break

    # Snap to half-integers and re-verify; if it breaks, keep the float version.
    Uf, Vf, Wf = best
    U2, V2, W2 = _snaps_half(Uf, Vf, Wf)
    if _verify(U2, V2, W2, N, M, P, tol=1e-6):
        Uf, Vf, Wf = U2, V2, W2

    U_out = [[float(x) for x in row] for row in Uf]
    V_out = [[float(x) for x in row] for row in Vf]
    W_out = [[float(x) for x in row] for row in Wf]
    return {"U": U_out, "V": V_out, "W": W_out}
