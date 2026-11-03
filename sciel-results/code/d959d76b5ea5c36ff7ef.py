"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Fast tensor verification: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0])
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    max_err = 0.0
    for c in range(rank):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            row = wc[:, i]                 # (p,)
            for j in range(m):
                max_err = max(max_err, abs(np.dot(uc[i, j] * vc[j, :], row) - 1.0))
    return max_err < tol


def _linear_W(U, V, n, m, p):
    """Given U,V, solve the linear system for W so the decomposition is exact (if consistent)."""
    rank = len(U[0])
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    A = np.zeros((n * m * p, rank))
    b = np.ones(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    A[idx, c] = Ua[i * m + j, c] * Va[j * p + k, c]
                idx += 1
    try:
        Ws, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    if not np.allclose(A @ Ws, b, atol=1e-7, rtol=1e-7):
        return None
    return Ws


def _polish(U, V, W, n, m, p, max_iter=80):
    """Exact-rational least-squares refinement (change of variables to remove the
    integer/half-integer constraint, then a high-precision exact local solve)."""
    rank = len(U[0])
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    # Build linear map: vec(T) = M @ vec(W)
    M = np.zeros((n * m * p, n * p * rank))
    for c in range(rank):
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    M[(i * m + j) * p + k, c * (n * p) + k * n + i] = (
                        Ua[i * m + j, c] * Va[j * p + k, c])
    b = np.ones(n * m * p)
    wvec = Wf.reshape(-1)
    for _ in range(max_iter):
        try:
            coef, res, rk, sv = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            break
        err = np.linalg.norm(M @ coef - b)
        if err < 1e-9:
            wvec = coef
            break
        wvec = coef
        Wf = wvec.reshape(rank, n * p)
        # rationalize (denominator 2) -> exact half-integers
        Wr = np.round(Wf * 2.0) / 2.0
        # Gauss-Newton step on U,V
        for _ in range(2):
            J = np.zeros((n * m * p, n * m * rank + m * p * rank))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i * m + j) * p + k
                            J[row, c * (n * m) + i * m + j] = Va[j * p + k, c] * Wr[c, k * n + i]
                            J[row, (n * m * rank) + c * (m * p) + j * p + k] = Ua[i * m + j, c] * Wr[c, k * n + i]
            try:
                step, *_ = np.linalg.lstsq(J, M @ wvec - b, rcond=None)
            except Exception:
                step = np.zeros(J.shape[1])
            if np.linalg.norm(step) < 1e-12:
                break
            uvec = Ua.reshape(-1) - step[:n * m * rank]
            vvec = Va.reshape(-1) - step[n * m * rank:]
            Ur = np.round(uvec.reshape(n * m, rank) * 2.0) / 2.0
            Vr = np.round(vvec.reshape(m * p, rank) * 2.0) / 2.0
            Ua = Ur
            Va = Vr
    Uo = np.round(Ua * 2.0) / 2.0
    Vo = np.round(Va * 2.0) / 2.0
    Wo = np.round(Wf * 2.0) / 2.0
    return Uo.tolist(), Vo.tolist(), Wo.tolist()


def _try_rank(U, V, n, m, p, deadline):
    """Given U,V (rank r), produce an exact decomposition if one exists."""
    Ws = _linear_W(U, V, n, m, p)
    if Ws is None:
        return None
    Uo = np.round(np.asarray(U, dtype=float) * 2.0) / 2.0
    Vo = np.round(np.asarray(V, dtype=float) * 2.0) / 2.0
    Wo = np.round(Ws * 2.0) / 2.0
    if _verify(Uo.tolist(), Vo.tolist(), Wo.tolist(), n, m, p):
        return Uo.tolist(), Vo.tolist(), Wo.tolist()
    if time.time() < deadline:
        U2, V2, W2 = _polish(Uo.tolist(), Vo.tolist(), Wo.tolist(), n, m, p)
        if _verify(U2, V2, W2, n, m, p):
            return U2, V2, W2
    return None


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # ---- Phase 1: structured "row/column" factorization (fast, exact) ----
    # U[i*m+j,c]=u_i * a_j, V[j*p+k,c]=a_j * v_k, W[k*n+i,c]=w_{k,i}
    # T[i,j,k]=sum_c u_i*a_j^2*v_k*w_{k,i}=1  =>  M[i,k]=sum_c u_i*v_k*w_{k,i}=1
    # so we need a rank-1 M = u * w^T  (u_i*v_k*w_{k,i}=1).
    for u in [[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [1, 1, 0.5]]:
        if time.time() >= deadline:
            break
        for v in [[1, 1, 1], [1, 1, 2], [2, 1, 1], [1, 2, 2], [1, 0.5, 1]]:
            if time.time() >= deadline:
                break
            u = np.array(u, float)
            v = np.array(v, float)
            w = np.ones((p, n))
            for k in range(p):
                for i in range(n):
                    w[k, i] = 1.0 / (u[i] * v[k])
            # build rank-r decomposition of M = u*w^T with r = number of distinct columns of w
            cols = {}
            for k in range(p):
                key = tuple(np.round(w[k] * 2.0) / 2.0)
                cols.setdefault(key, []).append(k)
            keys = list(cols.keys())
            r = len(keys)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for ci, key in enumerate(keys):
                wv = np.array(key)
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, ci] = u[i]
                    V[j * p + k, ci] = 1.0
                    for k in range(p):
                        W[k * n + i, ci] = 0.0
                for k in cols[key]:
                    for j in range(m):
                        V[j * p + k, ci] = 1.0
                    for i in range(n):
                        W[k * n + i, ci] = wv[i]
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p) and r < best_rank:
                best_rank = r
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Phase 2: random U,V + exact linear W + rational polish ----
    rng = random.Random(12345)
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
    max_rank = min(best_rank - 1, 60)
    for rank in range(24, max_rank + 1):
        if time.time() >= deadline - 5:
            break
        budget = deadline - 20
        attempts = 0
        while time.time() < budget and attempts < 4000:
            attempts += 1
            U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m * p)])
            res = _try_rank(U, V, n, m, p, deadline)
            if res is not None:
                Uo, Vo, Wo = res
                if rank < best_rank:
                    best_rank = rank
                    best = {"U": Uo, "V": Vo, "W": Wo}
                    break  # found at this rank; try lower next

    # ---- Phase 3: cancellation / rank reduction of best found ----
    if time.time() < deadline:
        Ua = np.asarray(best["U"], float)
        Va = np.asarray(best["V"], float)
        Wa = np.asarray(best["W"], float)
        r = Ua.shape[1]
        for c_remove in range(r):
            if time.time() >= deadline:
                break
            U2 = np.delete(Ua, c_remove, axis=1)
            V2 = np.delete(Va, c_remove, axis=1)
            W2 = np.delete(Wa, c_remove, axis=1)
            if _verify(U2.tolist(), V2.tolist(), W2.tolist(), n, m, p):
                best_rank = r - 1
                best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                break

    return best
# EVOLVE-BLOCK-END
