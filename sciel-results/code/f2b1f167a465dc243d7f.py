"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def build_block_decomp():
        U_strassen = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_strassen = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W_strassen = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        U_school = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        V_school = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        W_school = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        total_rank = 7 * 7 + 4
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        col_offset = 0
        for b in range(7):
            k_start = 2 * b
            U[:, col_offset:col_offset+7] = U_strassen
            for j in range(m):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_strassen[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_strassen[k_local * 2 + i, :]
            col_offset += 7
        k_start = 14
        U[:, col_offset:col_offset+4] = U_school
        for j in range(m):
            V[j * p + k_start, col_offset:col_offset+4] = V_school[j, :]
        for i in range(n):
            W[k_start * n + i, col_offset:col_offset+4] = W_school[i, :]
        return U, V, W

    def residual_flat(x, R):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        res = np.empty(n*p*m*m)
        idx = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.dot(W[k*n+i, :], U[i*m+j, :] * V[jp*p+k, :])
                        target = 1.0 if j == jp else 0.0
                        res[idx] = val - target
                        idx += 1
        return res

    def try_rank_lm(R, x0, max_nfev=2000):
        def res_func(x):
            return residual_flat(x, R)
        try:
            result = least_squares(res_func, x0, method='lm', max_nfev=max_nfev,
                                   xtol=1e-14, ftol=1e-14, gtol=1e-14)
            max_res = np.max(np.abs(result.fun))
            return result.x, max_res
        except Exception:
            return None, np.inf

    def verify(U, V, W, R, tol=1e-8):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.dot(W[k*n+i, :], U[i*m+j, :] * V[jp*p+k, :])
                        target = 1.0 if j == jp else 0.0
                        if abs(val - target) > tol:
                            return False
        return True

    # Baseline
    U_base, V_base, W_base = build_block_decomp()
    best_rank = 53
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()

    # Try lower ranks with LM optimization
    np.random.seed(42)
    for R in range(52, 39, -1):
        if time.time() > deadline - 20:
            break
        found = False
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            # Initialize: scale down baseline truncated + random perturbation
            if R <= 53:
                U0 = U_base[:, :R].copy()
                V0 = V_base[:, :R].copy()
                W0 = W_base[:, :R].copy()
            else:
                U0 = np.random.randn(n*m, R) * 0.5
                V0 = np.random.randn(m*p, R) * 0.5
                W0 = np.random.randn(p*n, R) * 0.5
            # Add small random perturbation to escape local minima
            U0 += np.random.randn(n*m, R) * 0.01
            V0 += np.random.randn(m*p, R) * 0.01
            W0 += np.random.randn(p*n, R) * 0.01
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x_opt, max_res = try_rank_lm(R, x0)
            if x_opt is not None and max_res < 1e-10:
                U_opt = x_opt[:n*m*R].reshape(n*m, R)
                V_opt = x_opt[n*m*R:2*n*m*R].reshape(m*p, R)
                W_opt = x_opt[2*n*m*R:].reshape(p*n, R)
                if verify(U_opt, V_opt, W_opt, R):
                    best_rank = R
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    break
        if found:
            break

    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 53

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
