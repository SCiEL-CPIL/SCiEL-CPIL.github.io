"""Matrix multiplication 2x3x7 low-rank exact decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _t0():
    return time.time()

def _elapsed(t0):
    return time.time() - t0

def _to_frac_mat(M):
    out = []
    for row in M:
        r = []
        for x in row:
            hx = round(float(x) * 2) / 2.0
            r.append(Fraction(hx))
        out.append(r)
    return out

def _exact_check(Uf, Vf, Wf, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(len(Uf[0])):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _make_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    t0 = _t0()
    deadline = t0 + 135.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n*m*p

    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = schoolbook_rank

    def tensor_from(U, V, W):
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        return T

    def try_refine(U, V, W, r, max_time):
        nonlocal best_U, best_V, best_W, best_rank
        start = time.time()
        # Iterative refinement
        for it in range(5000):
            if time.time() - start > max_time or _elapsed(t0) > deadline:
                break
            Tcur = tensor_from(U, V, W)
            resid = 1.0 - Tcur
            if np.max(np.abs(resid)) < 1e-10:
                break
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_ = resid[i,j,k]
                        dU[i*m+j] += r_ * (V[j*p+k] * W[k*n+i])
                        dV[j*p+k] += r_ * (U[i*m+j] * W[k*n+i])
                        dW[k*n+i] += r_ * (U[i*m+j] * V[j*p+k])
            lr = 0.05
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        Tcur = tensor_from(U, V, W)
        err = np.max(np.abs(Tcur - 1.0))
        if err < 1e-4:
            Us = np.round(U * 2) / 2.0
            Vs = np.round(V * 2) / 2.0
            Ws = np.round(W * 2) / 2.0
            Uf = _to_frac_mat(Us)
            Vf = _to_frac_mat(Vs)
            Wf = _to_frac_mat(Ws)
            if _exact_check(Uf, Vf, Wf, n, m, p):
                best_U, best_V, best_W = Us, Vs, Ws
                best_rank = r
                return True
        return False

    # Try to reduce rank by merging schoolbook terms
    # For each pair of terms, check if they can be combined
    # This is complex; instead, try random restarts at lower ranks
    import random
    rng = np.random.RandomState(12345)

    # Try ranks from 35 down to 10, limited attempts each
    for r in range(min(schoolbook_rank - 1, 35), 9, -1):
        if _elapsed(t0) > deadline:
            break
        if r >= best_rank:
            continue
        # A few random initializations
        for attempt in range(3):
            if _elapsed(t0) > deadline:
                break
            U = rng.randn(nm, r) * 0.3
            V = rng.randn(mp, r) * 0.3
            W = rng.randn(pn, r) * 0.3
            # Ensure reasonable scale
            try_refine(U, V, W, r, max_time=3.0)
            if best_rank <= r:
                break

    # Final verification
    Uf = _to_frac_mat(best_U)
    Vf = _to_frac_mat(best_V)
    Wf = _to_frac_mat(best_W)
    if not _exact_check(Uf, Vf, Wf, n, m, p):
        Uf, Vf, Wf = _to_frac_mat(U0), _to_frac_mat(V0), _to_frac_mat(W0)

    return {"U": Uf, "V": Vf, "W": Wf}
# EVOLVE-BLOCK-END
