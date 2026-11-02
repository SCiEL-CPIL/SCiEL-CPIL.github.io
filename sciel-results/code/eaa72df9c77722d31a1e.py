"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R = 15  # target rank for 2x3x3

    def residual(params):
        U = params[:nm * R].reshape(nm, R)
        V = params[nm * R: nm * R + mp * R].reshape(mp, R)
        W = params[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]) - 1.0
                    idx += 1
        return res

    def fallback():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    deadline = time.time() + 140
    rng = np.random.default_rng(42)
    best_params = None
    best_cost = np.inf

    for attempt in range(40):
        if time.time() > deadline:
            break
        params = rng.standard_normal(nm * R + mp * R + pn * R) * 0.5
        try:
            result = least_squares(residual, params, method='lm', max_nfev=5000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
            if result.cost < 1e-14:
                break
        except Exception:
            pass

    if best_params is None or best_cost > 1e-8:
        return fallback()

    U = best_params[:nm * R].reshape(nm, R)
    V = best_params[nm * R: nm * R + mp * R].reshape(mp, R)
    W = best_params[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)

    # Verify exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-8:
                    return fallback()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
