import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a,b,c] = 1.0
    diff = np.abs(T - target)
    return np.max(diff) < tol

def construct():
    start = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    Na, Nb, Nc = n*m, m*p, p*n
    
    target = np.zeros((Na, Nb, Nc))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n*m*p

    def try_rank(r, seeds=50, iters=50):
        for seed in range(seeds):
            if time.time() > deadline - 10:
                return False, None, None, None
            rng = np.random.RandomState(seed)
            U = rng.randn(Na, r)
            V = rng.randn(Nb, r)
            W = rng.randn(Nc, r)
            
            for _ in range(iters):
                VW = np.kron(V, W)
                T0 = target.reshape(Na, Nb*Nc)
                U, _, _ = np.linalg.lstsq(VW.T, T0.T, rcond=None)
                
                WU = np.kron(W, U)
                T1 = target.transpose(1,2,0).reshape(Nb, Nc*Na)
                V, _, _ = np.linalg.lstsq(WU.T, T1.T, rcond=None)
                
                UV = np.kron(U, V)
                T2 = target.transpose(2,0,1).reshape(Nc, Na*Nb)
                W, _, _ = np.linalg.lstsq(UV.T, T2.T, rcond=None)
                
            Ur = np.round(U * 2) / 2.0
            Vr = np.round(V * 2) / 2.0
            Wr = np.round(W * 2) / 2.0
            
            if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, tol=1e-6):
                return True, Ur, Vr, Wr
        return False, None, None, None

    # Try ranks from 32 up to 35
    for r in range(32, best_rank):
        if time.time() > deadline - 15:
            break
        found, U, V, W = try_rank(r, seeds=100, iters=30)
        if found:
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            break

    def to_clean(mat):
        out = []
        for row in mat:
            nr = []
            for v in row:
                if isinstance(v, float):
                    if abs(v - round(v)) < 1e-9:
                        nr.append(int(round(v)))
                    else:
                        nr.append(v)
                else:
                    nr.append(v)
            out.append(nr)
        return out

    best["U"] = to_clean(best["U"])
    best["V"] = to_clean(best["V"])
    best["W"] = to_clean(best["W"])
    return best
