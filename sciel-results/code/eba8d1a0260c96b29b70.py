import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i*m+j, j*p+k, k*n+i))
    
    nm, mp, pn = n*m, m*p, p*n
    n_entries = len(entries)
    
    for rank in range(15, n*m*p + 1):
        if time.time() > deadline:
            break
        result = _search_rank(entries, nm, mp, pn, rank, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank(entries, nm, mp, pn, rank, deadline):
    n_entries = len(entries)
    total_vars = nm*rank + mp*rank + pn*rank
    
    def residual(x):
        if time.time() > deadline:
            raise ValueError("timeout")
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        res = np.empty(n_entries)
        for idx, (a,b,c) in enumerate(entries):
            res[idx] = np.dot(U[a,:], V[b,:] * W[c,:]) - 1
        return res
    
    for seed in range(5):
        if time.time() > deadline:
            return None
        np.random.seed(seed * 10000 + rank)
        x0 = np.random.randn(total_vars) * 0.5
        try:
            result = least_squares(residual, x0, method='lm', max_nfev=500)
        except:
            continue
        if np.max(np.abs(result.fun)) < 1e-8:
            x = result.x
            U = x[:nm*rank].reshape(nm, rank)
            V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = x[nm*rank+mp*rank:].reshape(pn, rank)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            ok = True
            for (a,b,c) in entries:
                if abs(np.dot(U_r[a,:], V_r[b,:] * W_r[c,:]) - 1) > 1e-9:
                    ok = False
                    break
            if ok:
                return (U_r, V_r, W_r)
    return None
