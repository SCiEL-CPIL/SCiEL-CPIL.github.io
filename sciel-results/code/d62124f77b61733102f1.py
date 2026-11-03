"""Search for a rank-minimal exact decomposition of the 3x4 by 4x6 matrix-multiplication tensor."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS

# EVOLVE-BLOCK-START

def _build_T():
    """Matrix-multiplication tensor T[i,j,k] = 1 (trace convention: W[k*n+i])."""
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0
    return T


def _schoolbook():
    """Rank n*m*p = 72 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W


def _reconstruct(U, V, W):
    """Reconstruct the tensor from factor matrices."""
    return np.einsum("ic,jc,kc->ijk", U, V, W)


def _err(U, V, W, T):
    R = _reconstruct(U, V, W)
    return np.linalg.norm(R - T)


def _als_step(U, V, W, T, iters=80, lam=1e-6):
    """Alternating least squares on the smooth Frobenius objective."""
    r = U.shape[1]
    for _ in range(iters):
        # U
        VT = np.einsum("jc,kc->jkc", V, W)          # (M,P,N)
        Tt = np.transpose(T, (1, 2, 0))             # (M,P,N)
        U = np.linalg.lstsq(VT, Tt, rcond=None)[0]  # (M*N, r)
        # V
        UT = np.einsum("ic,kc->ikc", U, W)          # (N,P,M)
        Tt = np.transpose(T, (2, 0, 1))             # (P,M,N)
        V = np.linalg.lstsq(UT, Tt, rcond=None)[0]  # (M*P, r)
        # W
        UV = np.einsum("ic,jc->ijc", U, V)          # (N,M,P)
        W = np.linalg.lstsq(UV, T, rcond=None)[0]   # (P*N, r)
    return U, V, W


def _round_half(U, V, W):
    return np.round(2.0 * U) / 2.0, np.round(2.0 * V) / 2.0, np.round(2.0 * W) / 2.0


def _is_exact(U, V, W, T, tol=1e-6):
    R = _reconstruct(U, V, W)
    return float(np.max(np.abs(R - T))) <= tol


def _try_rank(T, seed, deadline):
    """For a given target rank, run seeds + ALS + exact rounding. Return best exact (U,V,W) or None."""
    best = None
    best_err = math.inf
    nseeds = 6
    for s in range(nseeds):
        if time.time() > deadline:
            break
        # seed: take schoolbook, drop (rank-72) random terms, merge each dropped term into a random kept term
        U0, V0, W0 = _schoolbook()
        r0 = U0.shape[1]
        perm = list(range(r0))
        random.shuffle(perm)
        keep = perm[:seed]
        drop = perm[seed:]
        U = U0[:, keep].copy()
        V = V0[:, keep].copy()
        W = W0[:, keep].copy()
        # merge each dropped term into a random kept term (same (i,j,k) support => additive)
        for c in drop:
            if not keep:
                break
            t = keep[random.randrange(len(keep))]
            U[:, t] += U[:, c]
            V[:, t] += V[:, c]
            W[:, t] += W[:, c]
        # small random perturbation to escape the schoolbook basin
        U = U + 1e-3 * np.random.randn(*U.shape)
        V = V + 1e-3 * np.random.randn(*V.shape)
        W = W + 1e-3 * np.random.randn(*W.shape)
        U, V, W = _als_step(U, V, W, T, iters=60)
        err = _err(U, V, W, T)
        if err < best_err:
            best_err = err
        # exact rounding + check
        Uh, Vh, Wh = _round_half(U, V, W)
        if _is_exact(Uh, Vh, Wh, T):
            if best is None or Uh.shape[1] < best[0].shape[1]:
                best = (Uh, Vh, Wh)
        if best is not None:
            break
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 110.0   # safety margin before the 150s wall-clock kill
    random.seed(12345)
    np.random.seed(12345)
    T = _build_T()
    # guaranteed fallback
    U, V, W = _schoolbook()
    best = (U, V, W)
    # try strictly lower ranks first (71, 70, ...)
    for target in range(N * M * P - 1, max(N * M * P - 6, 0), -1):
        if time.time() > deadline:
            break
        res = _try_rank(T, target, deadline)
        if res is not None:
            best = res
            break
    # final safety: ensure exactness of whatever we return
    if not _is_exact(best[0], best[1], best[2], T):
        U, V, W = _schoolbook()
        best = (U, V, W)
    if time.time() > deadline - 5.0:
        U, V, W = _schoolbook()
        best = (U, V, W)
    U, V, W = best
    # convert to plain lists (floats are fine for integer/half-integer entries)
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }

# EVOLVE-BLOCK-END
