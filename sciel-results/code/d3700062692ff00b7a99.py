import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 110

    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        for i in range(2):
            for k in range(5):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        def strassen(c, i_off, j_off, k_off):
            U[i_off * m + j_off, c] = 1
            U[(i_off + 1) * m + j_off + 1, c] = 1
            V[j_off * p + k_off, c] = 1
            V[(j_off + 1) * p + k_off + 1, c] = 1
            W[k_off * n + i_off, c] = 1
            W[(k_off + 1) * n + i_off + 1, c] = 1
            c += 1
            U[(i_off + 1) * m + j_off, c] = 1
            U[(i_off + 1) * m + j_off + 1, c] = 1
            V[j_off * p + k_off, c] = 1
            W[k_off * n + i_off + 1, c] = 1
            W[(k_off + 1) * n + i_off + 1, c] = -1
            c += 1
            U[i_off * m + j_off, c] = 1
            V[j_off * p + k_off + 1, c] = 1
            V[(j_off + 1) * p + k_off + 1, c] = -1
            W[(k_off + 1) * n + i_off, c] = 1
            W[(k_off + 1) * n + i_off + 1, c] = 1
            c += 1
            U[(i_off + 1) * m + j_off + 1, c] = 1
            V[(j_off + 1) * p + k_off, c] = 1
            V[j_off * p + k_off, c] = -1
            W[k_off * n + i_off, c] = 1
            W[k_off * n + i_off + 1, c] = 1
            c += 1
            U[i_off * m + j_off, c] = 1
            U[i_off * m + j_off + 1, c] = 1
            V[(j_off + 1) * p + k_off + 1, c] = 1
            W[k_off * n + i_off, c] = -1
            W[(k_off + 1) * n + i_off, c] = 1
            c += 1
            U[(i_off + 1) * m + j_off, c] = 1
            U[i_off * m + j_off, c] = -1
            V[j_off * p + k_off, c] = 1
            V[j_off * p + k_off + 1, c] = 1
            W[(k_off + 1) * n + i_off + 1, c] = 1
            c += 1
            U[i_off * m + j_off + 1, c] = 1
            U[(i_off + 1) * m + j_off + 1, c] = -1
            V[(j_off + 1) * p + k_off, c] = 1
            V[(j_off + 1) * p + k_off + 1, c] = 1
            W[k_off * n + i_off, c] = 1
            c += 1
            return c

        col = strassen(col, 0, 0, 0)
        col = strassen(col, 0, 0, 2)
        U[0, col] = 1; V[4, col] = 1; W[8, col] = 1; col += 1
        U[1, col] = 1; V[9, col] = 1; W[8, col] = 1; col += 1
        U[3, col] = 1; V[4, col] = 1; W[9, col] = 1; col += 1
        U[4, col] = 1; V[9, col] = 1; W[9, col] = 1; col += 1
        assert col == rank
        return U, V, W

    U28, V28, W28 = build_rank_28()

    def verify(U, V, W, rank, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :rank], V[j * p + k, :rank] * W[k * n + i, :rank])
                    if abs(val - 1) > tol:
                        return False
        return True

    def to_halfint(x):
        return np.round(x * 2) / 2.0

    # Try rank 27 with a few random restarts
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        rank = 27
        np.random.seed(42 + trial)
        U = U28[:, :rank].copy() + np.random.randn(n * m, rank) * 0.3
        V = V28[:, :rank].copy() + np.random.randn(m * p, rank) * 0.3
        W = W28[:, :rank].copy() + np.random.randn(p * n, rank) * 0.3
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def objective(x):
            Ux = x[:rank * n * m].reshape(n * m, rank)
            Vx = x[rank * n * m:rank * (n * m + m * p)].reshape(m * p, rank)
            Wx = x[rank * (n * m + m * p):].reshape(p * n, rank)
            res = np.array([
                np.sum(Ux[i * m + j] * Vx[j * p + k] * Wx[k * n + i]) - 1
                for i in range(n) for j in range(m) for k in range(p)
            ])
            return np.sum(res * res)

        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 50, 'maxfun': 200})
            if res.fun < 1e-10:
                Uf = res.x[:rank * n * m].reshape(n * m, rank)
                Vf = res.x[rank * n * m:rank * (n * m + m * p)].reshape(m * p, rank)
                Wf = res.x[rank * (n * m + m * p):].reshape(p * n, rank)
                Ur, Vr, Wr = to_halfint(Uf), to_halfint(Vf), to_halfint(Wf)
                if verify(Ur, Vr, Wr, rank, tol=1e-6):
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        except Exception:
            pass

    # Try rank 26 briefly
    if time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            rank = 26
            np.random.seed(100 + trial)
            U = U28[:, :rank].copy() + np.random.randn(n * m, rank) * 0.3
            V = V28[:, :rank].copy() + np.random.randn(m * p, rank) * 0.3
            W = W28[:, :rank].copy() + np.random.randn(p * n, rank) * 0.3
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def objective26(x):
                Ux = x[:rank * n * m].reshape(n * m, rank)
                Vx = x[rank * n * m:rank * (n * m + m * p)].reshape(m * p, rank)
                Wx = x[rank * (n * m + m * p):].reshape(p * n, rank)
                res = np.array([
                    np.sum(Ux[i * m + j] * Vx[j * p + k] * Wx[k * n + i]) - 1
                    for i in range(n) for j in range(m) for k in range(p)
                ])
                return np.sum(res * res)

            try:
                res = minimize(objective26, x0, method='L-BFGS-B',
                               options={'maxiter': 30, 'maxfun': 100})
                if res.fun < 1e-10:
                    Uf = res.x[:rank * n * m].reshape(n * m, rank)
                    Vf = res.x[rank * n * m:rank * (n * m + m * p)].reshape(m * p, rank)
                    Wf = res.x[rank * (n * m + m * p):].reshape(p * n, rank)
                    Ur, Vr, Wr = to_halfint(Uf), to_halfint(Vf), to_halfint(Wf)
                    if verify(Ur, Vr, Wr, rank, tol=1e-6):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
            except Exception:
                pass

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
