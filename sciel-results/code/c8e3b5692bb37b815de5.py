"""Find rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_rec - T)) < tol

def _try_rank(rank, T, max_iters=3000, seed=42):
    np.random.seed(seed)
    n, m, p = DIMENSIONS
    du, dv, dw = n*m, m*p, p*n
    U = np.random.randn(du, rank) * 0.3
    V = np.random.randn(dv, rank) * 0.3
    W = np.random.randn(dw, rank) * 0.3
    best_res = np.inf
    best = (U.copy(), V.copy(), W.copy())
    for it in range(max_iters):
        # Update U (fix V, W)
        M = np.zeros((rank, dv * dw))
        for r in range(rank):
            M[r] = np.outer(V[:, r], W[:, r]).flatten()
        Tf = T.reshape(du, dv * dw)
        try:
            U = np.linalg.lstsq(M.T, Tf.T, rcond=None)[0].T
        except: pass
        # Update V (fix U, W)
        M2 = np.zeros((rank, du * dw))
        for r in range(rank):
            M2[r] = np.outer(U[:, r], W[:, r]).flatten()
        Tf2 = T.reshape(du, dv, dw).transpose(1,0,2).reshape(dv, du*dw)
        try:
            V = np.linalg.lstsq(M2.T, Tf2.T, rcond=None)[0].T
        except: pass
        # Update W (fix U, V)
        M3 = np.zeros((rank, du * dv))
        for r in range(rank):
            M3[r] = np.outer(U[:, r], V[:, r]).flatten()
        Tf3 = T.reshape(du, dv, dw).transpose(2,0,1).reshape(dw, du*dv)
        try:
            W = np.linalg.lstsq(M3.T, Tf3.T, rcond=None)[0].T
        except: pass
        res = np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
        if res < best_res:
            best_res = res
            best = (U.copy(), V.copy(), W.copy())
        if res < 1e-10:
            break
    if best_res < 1e-8:
        return best + (best_res,)
    return None

def _snap_half(mat):
    return np.round(mat * 2) / 2.0

def _to_frac(mat):
    rows, cols = mat.shape
    return [[Fraction(int(round(mat[i,j]*2)), 2) for j in range(cols)] for i in range(rows)]

def construct():
    t0 = time.time()
    deadline = 115.0
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Schoolbook fallback
    rank_sb = n*m*p
    Us = np.zeros((n*m, rank_sb)); Vs = np.zeros((m*p, rank_sb)); Ws = np.zeros((p*n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
    fallback = {"U": _to_frac(Us), "V": _to_frac(Vs), "W": _to_frac(Ws)}
    
    best = None
    # Try ranks from low to high
    for rank in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]:
        if time.time() - t0 > deadline - 15: break
        for seed in [42, 137, 256, 512, 1024, 2048, 4096, 8192]:
            if time.time() - t0 > deadline - 10: break
            res = _try_rank(rank, T, max_iters=1500, seed=seed)
            if res is not None:
                U, V, W, r = res
                Uh, Vh, Wh = _snap_half(U), _snap_half(V), _snap_half(W)
                if _verify(Uh, Vh, Wh, T, tol=1e-6):
                    best = {"U": _to_frac(Uh), "V": _to_frac(Vh), "W": _to_frac(Wh)}
                    break
        if best is not None:
            continue
    return best if best is not None else fallback

