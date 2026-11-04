import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54

    # Build schoolbook decomposition (rank 54) - always valid
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i * m + j, :] * V_mat[j * p + k, :], W_mat[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_half_int(arr):
        return np.round(arr * 2) / 2.0

    # Try to find a lower-rank decomposition using block structure
    # The 2x3x9 tensor can be viewed as 3 blocks of 2x3x3
    # Try to find shared structure by optimizing over the block structure

    # Approach: use rank-18 for each 2x3x3 block but try to merge
    # Actually, just try a few random restarts for direct 2x3x9 rank reduction

    for target_rank in [52, 50, 48]:
        if time.time() > deadline - 10:
            break

        np.random.seed(123)

        # Initialize with schoolbook + perturbation
        init_U = np.zeros((n * m, target_rank))
        init_V = np.zeros((m * p, target_rank))
        init_W = np.zeros((p * n, target_rank))

        for c in range(target_rank):
            init_U[:, c] = U[:, c]
            init_V[:, c] = V[:, c]
            init_W[:, c] = W[:, c]

        # Add small perturbation
        init_U += 0.01 * np.random.randn(n * m, target_rank)
        init_V += 0.01 * np.random.randn(m * p, target_rank)
        init_W += 0.01 * np.random.randn(p * n, target_rank)

        def objective(params):
            r = target_rank
            U_r = params[0:n*m*r].reshape(n*m, r)
            V_r = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_r = params[n*m*r + m*p*r:].reshape(p*n, r)

            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :] * V_r[j*p+k, :], W_r[k*n+i, :])
                        error += (val - 1.0)**2
            return error

        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])

        try:
            # Use gradient-based method for better convergence
            from scipy.optimize import minimize as scipy_minimize
            res = scipy_minimize(objective, x0, method='BFGS',
                               options={'maxiter': 100, 'gtol': 1e-8})
            if res.fun < 1e-6:
                U_r = res.x[0:n*m*target_rank].reshape(n*m, target_rank)
                V_r = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)

                U_snap = snap_half_int(U_r)
                V_snap = snap_half_int(V_r)
                W_snap = snap_half_int(W_r)

                if verify(U_snap, V_snap, W_snap):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    break
        except Exception:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
