import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, r

    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_rank = r_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    is_valid = True

    def tensor_err(U, V, W):
        e = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    e += (s - 1.0) ** 2
        return e

    def alt_minimize(U, V, W, max_iter=50000):
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    b = np.ones(p)
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    b = np.ones(n)
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    b = np.ones(m)
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            if it % 100 == 0:
                if tensor_err(U, V, W) < 1e-14:
                    break
        return tensor_err(U, V, W)

    def try_snap(U, V, W):
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if tensor_err(U2, V2, W2) < 1e-8:
            return U2, V2, W2, True
        return None, None, None, False

    for rank in range(24, r_sb + 1):
        if time.time() > deadline - 30:
            break
        for rs in range(15):
            if time.time() > deadline - 15:
                break
            scale = [0.3, 0.5, 1.0, 0.1, 2.0, 0.2, 1.5, 0.7, 0.4, 3.0, 0.8, 0.15, 0.6, 1.2, 0.25][rs]
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            err = alt_minimize(U, V, W)
            if err < 1e-10:
                U2, V2, W2, ok = try_snap(U, V, W)
                if ok:
                    best_rank, best_U, best_V, best_W, is_valid = rank, U2, V2, W2, True
                    break
                elif best_rank == r_sb:
                    best_rank, best_U, best_V, best_W, is_valid = rank, U, V, W, False
        if is_valid and best_rank < r_sb:
            break

    if not is_valid:
        best_U, best_V, best_W, best_rank = schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
