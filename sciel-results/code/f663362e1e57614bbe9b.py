"""Tensor rank search for 3x3x4 matrix multiplication (trace convention W[k*n+i, r])."""
import time
import random
import math
from fractions import Fraction

import numpy as np

# EVOLVE-BLOCK-START
DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_exact(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    T = _build_tensor()
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    recon = np.einsum('ir,js,kt->ijs', Ua, Va, Wa)
    return bool(np.allclose(recon, T, atol=tol, rtol=0.0))


def _rationalize(M, tol=1e-8):
    """Return list-of-lists of Fractions if entries are near rationals (small denom), else None."""
    if not M or not M[0]:
        return None
    out = []
    for row in M:
        orow = []
        for x in row:
            f = Fraction(x).limit_denominator(1000000)
            if abs(float(f) - x) > tol:
                return None
            orow.append(f)
        out.append(orow)
    return out


def _try_rank(r, T, max_iters=40, seed=None):
    n, m, p = DIMENSIONS
    I, J, K = n * m, m * p, p * n
    rng = random.Random(seed)
    best = None
    for it in range(max_iters):
        if _remaining() < 2.0:
            break
        U = np.random.randn(I, r)
        V = np.random.randn(J, r)
        W = np.random.randn(K, r)
        # Normalize columns to avoid scale imbalance
        for c in range(r):
            nU = np.linalg.norm(U[:, c]) or 1.0
            nV = np.linalg.norm(V[:, c]) or 1.0
            nW = np.linalg.norm(W[:, c]) or 1.0
            U[:, c] /= nU
            V[:, c] /= nV
            W[:, c] /= nW
        # Alternating least squares (a few passes)
        for _ in range(6):
            # Solve U: for each i, minimize || T[i] - V_i U_i W_i ||
            for i in range(I):
                # residual target: T[i] = sum_r U[i,r] V[:,r] W[:,r]^T
                # => U[i,:] = (V W^T)^{-1} T[i]
                B = V @ W.T  # (J,K)
                try:
                    U[i, :] = np.linalg.lstsq(B, T[i].ravel(), rcond=None)[0]
                except Exception:
                    pass
            for j in range(J):
                B = U @ W.T
                try:
                    V[j, :] = np.linalg.lstsq(B, T[:, j, :].ravel(), rcond=None)[0]
                except Exception:
                    pass
            for k in range(K):
                B = U @ V.T
                try:
                    W[k, :] = np.linalg.lstsq(B, T[:, :, k].ravel(), rcond=None)[0]
                except Exception:
                    pass
        # Evaluate
        recon = np.einsum('ir,js,kt->ijs', U, V, W)
        err = float(np.linalg.norm(recon - T))
        if best is None or err < best[0]:
            best = (err, U, V, W)
        if best[0] < 1e-7:
            break
    if best is None:
        return None
    err, U, V, W = best
    if err > 1e-6:
        return None
    # Convert to near-integers then rationalize
    Uf = np.round(U, 6)
    Vf = np.round(V, 6)
    Wf = np.round(W, 6)
    Ur = _rationalize(Uf.tolist(), tol=1e-6)
    Vr = _rationalize(Vf.tolist(), tol=1e-6)
    Wr = _rationalize(Wf.tolist(), tol=1e-6)
    if Ur is None or Vr is None or Wr is None:
        return None
    # Final exact check with Fractions via float (fast) then confirm
    if not _verify_exact([[float(x) for x in row] for row in Ur],
                         [[float(x) for x in row] for row in Vr],
                         [[float(x) for x in row] for row in Wr], tol=1e-9):
        return None
    return {"U": Ur, "V": Vr, "W": Wr}


def construct():
    global _START
    _START = time.time()
    fallback = _schoolbook()
    T = _build_tensor()
    n, m, p = DIMENSIONS
    # Try ranks below schoolbook in increasing order
    for r in range(1, n * m * p):
        if _remaining() < 5.0:
            break
        res = _try_rank(r, T, max_iters=30, seed=1234 + r)
        if res is not None:
            # Rationalize to plain ints/halves where possible
            return res
    return fallback
# EVOLVE-BLOCK-END
