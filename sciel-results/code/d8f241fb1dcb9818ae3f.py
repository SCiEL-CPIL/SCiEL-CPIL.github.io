import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS
N, M, P = n*m, m*p, p*n  # 6, 12, 8

# Precompute index mapping for the 24 tensor entries
_TRIPLES = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
_UROW = [i*m + j for (i, j, k) in _TRIPLES]
_VROW = [j*p + k for (i, j, k) in _TRIPLES]
_WROW = [k*n + i for (i, j, k) in _TRIPLES]
_NTR = len(_TRIPLES)  # 24

def _residual(x, r):
    U = x[:N*r].reshape(N, r)
    V = x[N*r:(N+M)*r].reshape(M, r)
    W = x[(N+M)*r:].reshape(P, r)
    res = np.empty(_NTR)
    for t in range(_NTR):
        res[t] = np.dot(U[_UROW[t]] * V[_VROW[t]], W[_WROW[t]]) - 1.0
    return res

def _jac(x, r):
    U = x[:N*r].reshape(N, r)
    V = x[N*r:(N+M)*r].reshape(M, r)
    W = x[(N+M)*r:].reshape(P, r)
    J = np.zeros((_NTR, (N+M+P)*r))
    for t in range(_NTR):
        ur, vr, wr = _UROW[t], _VROW[t], _WROW[t]
        row = t
        for c in range(r):
            J[row, ur*r + c] = V[vr, c] * W[wr, c]
            J[row, (N+vr)*r + c] = U[ur, c] * W[wr, c]
            J[row, (N+M+wr)*r + c] = U[ur, c] * V[vr, c]
    return J

def _verify(U, V, W):
    r = len(U[0])
    for t in range(_NTR):
        ur, vr, wr = _UROW[t], _VROW[t], _WROW[t]
        s = Fraction(0)
        for c in range(r):
            s += U[ur][c] * V[vr][c] * W[wr][c]
        if s != 1:
            return False
    return True

def _snap(A):
    return np.round(A * 2) / 2

def _to_frac(A):
    r = A.shape[1]
    return [[Fraction(A[a, b]) for b in range(r)] for a in range(A.shape[0])]

def _schoolbook():
    rank = n*m*p
    U = [[0]*rank for _ in range(N)]
    V = [[0]*rank for _ in range(M)]
    W = [[0]*rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    from scipy.optimize import least_squares
    r = 22
    total = (N+M+P)*r
    # Try many random restarts
    for seed in range(20000):
        if time.time() > deadline - 3.0:
            break
        rng = np.random.default_rng(seed)
        std = 1.0 if seed % 2 == 0 else 0.5
        x0 = rng.normal(0.0, std, size=total)
        try:
            sol = least_squares(_residual, x0, args=(r,), jac=_jac,
                                method='lm', max_nfev=300,
                                ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except Exception:
            continue
        if sol.cost > 1e-4:
            continue
        x = sol.x
        U = x[:N*r].reshape(N, r)
        V = x[N*r:(N+M)*r].reshape(M, r)
        W = x[(N+M)*r:].reshape(P, r)
        Us, Vs, Ws = _snap(U), _snap(V), _snap(W)
        Uf, Vf, Wf = _to_frac(Us), _to_frac(Vs), _to_frac(Ws)
        if _verify(Uf, Vf, Wf):
            # return as exact half-integer floats
            Uout = [[float(Us[a, b]) for b in range(r)] for a in range(N)]
            Vout = [[float(Vs[a, b]) for b in range(r)] for a in range(M)]
            Wout = [[float(Ws[a, b]) for b in range(r)] for a in range(P)]
            return {"U": Uout, "V": Vout, "W": Wout}
    return _schoolbook()
