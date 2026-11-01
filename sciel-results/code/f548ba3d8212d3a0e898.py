import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def compute_error(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((n * m, m * p, p * n))
        for s in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
        return np.sum((T_recon - T) ** 2), T_recon

    def is_valid(U, V, W, tol=1e-8):
        err, _ = compute_error(U, V, W)
        return err < tol

    def try_rank(r, n_restarts=300, max_iter=300):
        best_err = float('inf')
        best = None
        rng = np.random.RandomState(42)
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            lr = 0.1
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                err, T_recon = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                if err < 1e-14:
                    break
                diff = T_recon - T
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for s in range(r):
                    dU[:, s] = np.einsum('abc,bc->a', diff, V[:, s] * W[:, s])
                    dV[:, s] = np.einsum('abc,ac->b', diff, U[:, s] * W[:, s])
                    dW[:, s] = np.einsum('abc,ab->c', diff, U[:, s] * V[:, s])
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                lr *= 0.999
        return best_err, best

    for r in [10, 9, 8, 7, 6]:
        if time.time() > deadline - 25:
            break
        err, best = try_rank(r)
        if best is not None and err < 1e-10:
            U, V, W = best
            Ur, Vr, Wr = np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2
            if is_valid(Ur, Vr, Wr):
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
            if is_valid(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
