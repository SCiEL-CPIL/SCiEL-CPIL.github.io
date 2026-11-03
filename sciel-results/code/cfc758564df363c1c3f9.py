"""Structural redesign: direct rank-R tensor decomposition via randomized
coordinate-descent with a hard wall-clock deadline guard.

Optimize U (n*m x R), V (m*p x R), W (p*n x R) so that
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i, c]
exactly reproduces the 2x5x6 matrix-multiplication tensor, then quantize to
the nearest half-integer and verify exactness.  Falls back to the schoolbook
rank-60 decomposition if no smaller rank is found in time.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 130.0          # wall-clock safety margin (evaluator kills at 150s)
    n, m, p = DIMENSIONS

    a = n * m
    b = m * p
    c = p * n

    # Flat index tables for the three factor rows at each (i,j,k).
    iu = np.repeat(np.repeat(np.arange(n), m), p)
    ju = np.repeat(np.arange(m), p)
    ku = np.arange(p)
    u_idx = iu * m + ju
    v_idx = ju * p + ku
    w_idx = ku * n + iu
    N = n * m * p
    target = np.ones(N)

    def quantize(M):
        return np.rint(2.0 * M) / 2.0

    def verify(U, V, W, tol=1e-6):
        val = np.einsum("ic,jc,kc->c", U[u_idx], V[v_idx], W[w_idx])
        return np.max(np.abs(val - target)) <= tol

    def to_list(U, V, W):
        return (quantize(U).tolist(), quantize(V).tolist(), quantize(W).tolist())

    # Deterministic fallback: schoolbook rank n*m*p (always valid).
    def schoolbook():
        R = n * m * p
        U = np.zeros((a, R)); V = np.zeros((b, R)); W = np.zeros((c, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return (U.tolist(), V.tolist(), W.tolist())

    best = schoolbook()
    best_rank = n * m * p
    rng = np.random.default_rng(12345)

    # Probe ranks in ascending order; the first valid hit is the smallest found.
    candidate_ranks = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    def try_rank(R):
        nonlocal best, best_rank
        while time.time() < DEADLINE - 1.0:
            U = rng.standard_normal((a, R)) * 0.5
            V = rng.standard_normal((b, R)) * 0.5
            W = rng.standard_normal((c, R)) * 0.5
            lr = 0.02
            for it in range(4000):
                if time.time() > DEADLINE - 1.0:
                    break
                val = np.einsum("ic,jc,kc->c", U[u_idx], V[v_idx], W[w_idx])
                err = target - val
                which = it % 3
                if which == 0:
                    M = V[v_idx] * W[w_idx]
                    gU = np.zeros((a, R))
                    np.add.at(gU, u_idx, -np.einsum("nr,cr->cr", M, err))
                    U[rng.integers(0, a)] -= lr * gU[rng.integers(0, a)]
                elif which == 1:
                    M = U[u_idx] * W[w_idx]
                    gV = np.zeros((b, R))
                    np.add.at(gV, v_idx, -np.einsum("nr,cr->cr", M, err))
                    V[rng.integers(0, b)] -= lr * gV[rng.integers(0, b)]
                else:
                    M = U[u_idx] * V[v_idx]
                    gW = np.zeros((c, R))
                    np.add.at(gW, w_idx, -np.einsum("nr,cr->cr", M, err))
                    W[rng.integers(0, c)] -= lr * gW[rng.integers(0, c)]
            Uq, Vq, Wq = to_list(U, V, W)
            if verify(np.array(Uq), np.array(Vq), np.array(Wq)):
                if R < best_rank:
                    best_rank = R
                    best = (Uq, Vq, Wq)
                return True
        return False

    for R in candidate_ranks:
        if time.time() > DEADLINE - 2.0:
            break
        if try_rank(R):
            break  # smallest valid rank found

    return {"U": best[0], "V": best[1], "W": best[2]}
# EVOLVE-BLOCK-END
