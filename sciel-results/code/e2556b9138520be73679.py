"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses ALS with multiple seeds, rank candidates, and a polish step.
    """
    start_time = time.time()
    deadline = 140  # 150s limit, 10s safety margin
    n, m, p = 3, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 12, 20, 15

    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def check_decomposition(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :rank], V[j * p + k, :rank] * W[k * n + i, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def als_optimize(rank, seed, max_iter=200):
        """
        Use ALS to find U, V, W of given rank.
        """
        rng = np.random.RandomState(seed)
        # Scale initialization to help convergence
        scale = 1.0 / np.sqrt(rank)
        U = rng.randn(nm, rank) * scale
        V = rng.randn(mp, rank) * scale
        W = rng.randn(pn, rank) * scale

        for it in range(max_iter):
            if time.time() - start_time > deadline:
                return None

            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass

            # Check error periodically
            if it % 20 == 0 or it == max_iter - 1:
                err = check_decomposition(U, V, W, rank)
                if err < 1e-10:
                    return U, V, W

        err = check_decomposition(U, V, W, rank)
        if err < 1e-10:
            return U, V, W
        return None

    def polish(U, V, W, rank, max_iter=50):
        """
        Final high-precision polish with tighter tolerance.
        """
        for it in range(max_iter):
            if time.time() - start_time > deadline:
                return None

            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass

            if it % 10 == 0 or it == max_iter - 1:
                err = check_decomposition(U, V, W, rank)
                if err < 1e-12:
                    return U, V, W

        err = check_decomposition(U, V, W, rank)
        if err < 1e-12:
            return U, V, W
        return None

    # Try ranks from low to high
    candidate_ranks = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_decomposition()

    for rank in candidate_ranks:
        if time.time() - start_time > deadline - 15:
            break

        found = False
        for seed in range(20):
            if time.time() - start_time > deadline - 15:
                break

            result = als_optimize(rank, seed, max_iter=150)
            if result is not None:
                U_opt, V_opt, W_opt = result
                # Try to polish
                polished = polish(U_opt, V_opt, W_opt, rank, max_iter=30)
                if polished is not None:
                    U_opt, V_opt, W_opt = polished
                # Final verification
                if check_decomposition(U_opt, V_opt, W_opt, rank) < 1e-10:
                    best_rank = rank
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    found = True
                    break

        if found:
            break

    # Convert to lists of Python floats
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
