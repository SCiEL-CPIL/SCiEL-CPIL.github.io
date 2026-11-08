"""Tensor-decomposition search for 4x4 x 4x5 matrix multiplication.

Returns a valid exact decomposition (T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]).
Lower rank is better. Schoolbook (rank 80) is the guaranteed valid fallback.
A hard wall-clock deadline guard keeps us safely inside the 150 s limit."""

import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
_START = time.time()
_DEADLINE = _START + 135.0   # safety margin before the 150 s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    for cand in _candidates(n, m, p, deadline=_DEADLINE):
        if time.time() > _DEADLINE - 4:
            break
        if cand is not None and len(cand["U"][0]) < len(best["U"][0]):
            best = cand
    return best
# EVOLVE-BLOCK-END


def _schoolbook(n, m, p):
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


def _tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            T[i, :, k] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    r = len(U[0])
    Ua = np.array(U, dtype=float).reshape(n, m, r)
    Va = np.array(V, dtype=float).reshape(m, p, r)
    Wa = np.array(W, dtype=float).reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    Tref = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            Tref[i, :, k] = 1.0
    return bool(np.allclose(T, Tref, atol=1e-9, rtol=0.0))


def _half(x):
    return int(round(x * 2)) / 2.0


def _rank1_fit(T, n, m, p, rng, iters=40):
    u = rng.standard_normal((n, m))
    v = rng.standard_normal((m, p))
    w = rng.standard_normal((p, n))
    u /= np.linalg.norm(u) + 1e-12
    v /= np.linalg.norm(v) + 1e-12
    w /= np.linalg.norm(w) + 1e-12
    for _ in range(iters):
        u_new = np.zeros((n, m))
        v_new = np.zeros((m, p))
        w_new = np.zeros((p, n))
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(p):
                    s += T[i, j, k] * v[j, k] * w[k, i]
                u_new[i, j] = s
        for j in range(m):
            for k in range(p):
                s = 0.0
                for i in range(n):
                    s += T[i, j, k] * u_new[i, j] * w[k, i]
                v_new[j, k] = s
        for k in range(p):
            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += T[i, j, k] * u_new[i, j] * v_new[j, k]
                w_new[k, i] = s
        u = u_new / (np.linalg.norm(u_new) + 1e-12)
        v = v_new / (np.linalg.norm(v_new) + 1e-12)
        w = w_new / (np.linalg.norm(w_new) + 1e-12)
    return u, v, w


def _candidates(n, m, p, deadline):
    T = _tensor(n, m, p)
    rng = random.Random(20240517)
    for target_rank in (56, 60, 64, 68, 72, 76, 80):
        if time.time() > deadline - 4:
            return
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        Tcur = T.copy()
        for c in range(target_rank):
            if time.time() > deadline - 4:
                return
            u, v, w = _rank1_fit(Tcur, n, m, p, rng)
            num = np.einsum('ijk,ij,jk,ki->', Tcur, u, v, w)
            den = np.einsum('ij,ij,jk,jk,ki,ki->', u, u, v, v, w, w)
            alpha = num / (den + 1e-12)
            U[:, c] = (alpha * u).reshape(n * m)
            V[:, c] = v.reshape(m * p)
            W[:, c] = w.reshape(p * n)
            Tcur = Tcur - alpha * np.einsum('ij,jk,ki->ijk', u, v, w)
            if np.linalg.norm(Tcur) < 1e-7:
                break
        Ur = [[_half(x) for x in row] for row in U]
        Vr = [[_half(x) for x in row] for row in V]
        Wr = [[_half(x) for x in row] for row in W]
        if _verify(Ur, Vr, Wr, n, m, p):
            yield {"U": Ur, "V": Vr, "W": Wr}
