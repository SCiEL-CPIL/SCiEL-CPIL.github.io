import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def verify(U, V, W, T, tol=1e-9):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def to_half_int(M):
    return np.round(np.array(M, dtype=float) * 2) / 2.0

def schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def als_decompose(T, rank, n, m, p, max_iter=1500, n_restarts=3, deadline=None):
    best_err = float('inf')
    best = None
    T_abc = T.reshape(n*m, m*p, p*n)
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            Q_all = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
            T_all = T_abc.transpose(0, 2, 1).reshape(-1, m*p)
            V = np.linalg.lstsq(Q_all, T_all, rcond=None)[0].T
            S_all = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            T2_all = T_abc.transpose(1, 2, 0).reshape(-1, n*m)
            U = np.linalg.lstsq(S_all, T2_all, rcond=None)[0].T
            R_all = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
            T3_all = T_abc.reshape(-1, p*n)
            W = np.linalg.lstsq(R_all, T3_all, rcond=None)[0].T
            if it % 200 == 199:
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.max(np.abs(recon - T)) > 0.5:
                    break
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < best_err:
            best_err = err
            best = (U, V, W)
    return best, best_err

def construct():
    start = time.time()
    deadline = start + 140
    n, m, p = DIMENSIONS
    T = build_target_tensor(n, m, p)
    U0, V0, W0 = schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n*m*p

    for rank in range(35, best_rank):
        if time.time() > deadline:
            break
        (U, V, W), err = als_decompose(T, rank, n, m, p,
                                       max_iter=1500, n_restarts=3, deadline=deadline)
        if err < 1e-5:
            Uh = to_half_int(U); Vh = to_half_int(V); Wh = to_half_int(W)
            if verify(Uh, Vh, Wh, T, tol=1e-7):
                best = (Uh, Vh, Wh)
                best_rank = rank
                break

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
