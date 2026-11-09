import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE_FRAC = 0.90
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)

def _make_2x2x2_strassen():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _make_4x4x4_strassen():
    n = m = p = 4
    R = 49
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    inner_U, inner_V, inner_W = _make_2x2x2_strassen()
    
    top_A = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    top_B = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    
    top_C = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    
    for t in range(7):
        for tp in range(7):
            col = t * 7 + tp
            
            for (br, bc, acoef) in top_A[t]:
                for ii in range(2):
                    for jj in range(2):
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        U[gi * m + gj, col] += acoef * inner_U[ii * 2 + jj, tp]
            
            for (bc2, bk, bcoef) in top_B[t]:
                for jj in range(2):
                    for kk in range(2):
                        gj = bc2 * 2 + jj
                        gk = bk * 2 + kk
                        V[gj * p + gk, col] += bcoef * inner_V[jj * 2 + kk, tp]
            
            for c_idx in range(4):
                if t in top_C[c_idx]:
                    cr, cc = c_idx // 2, c_idx % 2
                    for ii in range(2):
                        for kk in range(2):
                            gi = cr * 2 + ii
                            gk = cc * 2 + kk
                            W[gk * n + gi, col] += inner_W[kk * 2 + ii, tp]
    
    return U, V, W

def _strassen_444_5(n, m, p):
    U4, V4, W4 = _make_4x4x4_strassen()
    R4 = U4.shape[1]
    extra = n * m
    R = R4 + extra
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    U[:, :R4] = U4
    V[:, :R4] = V4
    W[:, :R4] = W4
    
    k = 4
    c = R4
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best = None
    
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_5(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
