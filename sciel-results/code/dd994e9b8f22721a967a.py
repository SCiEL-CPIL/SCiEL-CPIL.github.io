"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Try ranks from lower to upper; schoolbook is n*m*p = 60
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _try_rank(n, m, p, rank, deadline):
    max_restarts = 30
    max_iter = 200

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = np.random.RandomState(12345 + restart * 7919)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # --- Update U rows ---
            # For each (i, j): solve M^T * u = 1_p where M[k, r] = V[jp+k, r] * W[kn+i, r]
            for i in range(n):
                for j in range(m):
                    # Build p x rank matrix
                    M = np.empty((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update V rows ---
            # For each (j, k): solve N^T * v = 1_n where N[i, r] = U[im+j, r] * W[kn+i, r]
            for j in range(m):
                for k in range(p):
                    N_mat = np.empty((n, rank))
                    for i in range(n):
                        N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update W rows ---
            # For each (k, i): solve P^T * w = 1_m where P[j, r] = U[im+j, r] * V[jp+k, r]
            for k in range(p):
                for i in range(n):
                    P_mat = np.empty((m, rank))
                    for j in range(m):
                        P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # Check convergence every 10 iterations
            if it % 10 == 9:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-8:
                    result = _rationalize(U, V, W, n, m, p, rank)
                    if result is not None:
                        return result

    return None


def _rationalize(U, V, W, n, m, p, rank):
    """Try to snap to half-integer values and verify exact decomposition."""
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return None

    # Convert to lists of floats
    U_list = [[float(U_r[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V_r[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W_r[i, j]) for j in range(rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}


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
# EVOLVE-BLOCK-END
