"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from 59 down
    for rank in range(best_rank - 1, max(best_rank - 25, 10), -1):
        if time.time() > deadline - 15:
            break
        result = _try_rank_l_bfgs(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank <= 40:
                break

    return best_result


def _schoolbook(n, m, p):
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


def _try_rank_l_bfgs(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using L-BFGS-B optimization."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_vars = dim_u + dim_v + dim_w

    rng = random.Random(42 + rank * 13)

    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Random initialization with various scales
        scale = rng.uniform(0.5, 2.0)
        U = np.random.randn(dim_u, rank) * scale
        V = np.random.randn(dim_v, rank) * scale
        W = np.random.randn(dim_w, rank) * scale

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def objective(x):
            U_flat = x[:dim_u * rank].reshape(dim_u, rank)
            V_flat = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_flat = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)

            # Compute error: sum over all (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_flat[i * m + j, r] * V_flat[j * p + k, r] * W_flat[k * n + i, r]
                        err += (val - 1.0) ** 2
            return err

        def gradient(x):
            U_flat = x[:dim_u * rank].reshape(dim_u, rank)
            V_flat = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_flat = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)

            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_flat[i * m + j, r] * V_flat[j * p + k, r] * W_flat[k * n + i, r]
                        diff = 2.0 * (val - 1.0)
                        for r in range(rank):
                            grad_U[i * m + j, r] += diff * V_flat[j * p + k, r] * W_flat[k * n + i, r]
                            grad_V[j * p + k, r] += diff * U_flat[i * m + j, r] * W_flat[k * n + i, r]
                            grad_W[k * n + i, r] += diff * U_flat[i * m + j, r] * V_flat[j * p + k, r]

            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                                       'maxls': 50, 'maxfun': 1000})
        except Exception:
            continue

        if time.time() > deadline - 5:
            break

        if result.fun < 1e-10:
            U_opt = result.x[:dim_u * rank].reshape(dim_u, rank)
            V_opt = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W_opt = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)

            # Try to rationalize
            rat_result = _rationalize_optimized(U_opt, V_opt, W_opt, n, m, p)
            if rat_result is not None:
                return rat_result

            # If rationalization fails, try with more precision
            rat_result = _rationalize_optimized(U_opt, V_opt, W_opt, n, m, p, max_denom=16)
            if rat_result is not None:
                return rat_result

    return None


def _rationalize_optimized(U, V, W, n, m, p, max_denom=8):
    """Try to find rational/half-integer approximation to the optimized decomposition."""
    rank = U.shape[1]

    for denom in [1, 2, 4, 8, 16]:
        if denom > max_denom:
            break
        scale = 2.0 / denom

        U_int = np.round(U * scale).astype(int)
        V_int = np.round(V * scale).astype(int)
        W_int = np.round(W * scale).astype(int)

        target = scale ** 3
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err

        if max_err < 1e-6:
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]

            for i in range(n * m):
                for r in range(rank):
                    U_list[i][r] = U_int[i, r] / scale
            for i in range(m * p):
                for r in range(rank):
                    V_list[i][r] = V_int[i, r] / scale
            for i in range(p * n):
                for r in range(rank):
                    W_list[i][r] = W_int[i, r] / scale

            # Final verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                        if abs(val - 1.0) > 1e-9:
                            return None

            return {"U": U_list, "V": V_list, "W": W_list}

    return None
# EVOLVE-BLOCK-END
