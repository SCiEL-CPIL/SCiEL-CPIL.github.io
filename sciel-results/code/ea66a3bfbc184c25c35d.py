import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    time_limit = 115.0

    def try_rank(R, seed_base):
        if time.time() - start_time > time_limit:
            return None

        for restart in range(5):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(seed_base + restart * 1000 + R)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1

            lam = 1e-8
            I_R = np.eye(R) * lam

            for it in range(60):
                if time.time() - start_time > time_limit:
                    return None

                # Update U
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW + I_R
                R_U = np.einsum('vc,wc,uvw->uc', V, W, T, optimize=True)
                try:
                    U = np.linalg.solve(G_U, R_U.T).T
                except:
                    U = np.linalg.lstsq(G_U, R_U.T, rcond=None)[0].T

                # Update V
                UtU = U.T @ U
                G_V = UtU * WtW + I_R
                R_V = np.einsum('uc,wc,uvw->vc', U, W, T, optimize=True)
                try:
                    V = np.linalg.solve(G_V, R_V.T).T
                except:
                    V = np.linalg.lstsq(G_V, R_V.T, rcond=None)[0].T

                # Update W
                VtV = V.T @ V
                G_W = UtU * VtV + I_R
                R_W = np.einsum('uc,vc,uvw->wc', U, V, T, optimize=True)
                try:
                    W = np.linalg.solve(G_W, R_W.T).T
                except:
                    W = np.linalg.lstsq(G_W, R_W.T, rcond=None)[0].T

                if it % 20 == 19:
                    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-7:
                        # Try rounding
                        for mult in [2, 4, 8]:
                            U_r = np.round(U * mult) / mult
                            V_r = np.round(V * mult) / mult
                            W_r = np.round(W * mult) / mult
                            T_r = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r, optimize=True)
                            if np.linalg.norm(T_r - T) < 1e-9:
                                return (U_r, V_r, W_r)
                        # Try integer
                        U_r = np.round(U)
                        V_r = np.round(V)
                        W_r = np.round(W)
                        T_r = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r, optimize=True)
                        if np.linalg.norm(T_r - T) < 1e-9:
                            return (U_r, V_r, W_r)

        return None

    # Try ranks from 108 upward (Strassen-based estimate)
    for R in range(108, 130):
        result = try_rank(R, 42)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: schoolbook rank 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
