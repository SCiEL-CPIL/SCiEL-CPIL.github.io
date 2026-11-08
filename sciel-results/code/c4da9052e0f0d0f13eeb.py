import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill


def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 for all (i,j,k) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _compute_error(U, V, W, n, m, p):
    """
    Compute the error tensor: err[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    Returns numpy array of shape (n, m, p) and its norm.
    """
    rank = U.shape[1]
    err = np.ones((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err[i, j, k] -= u_r[i, j] * v_r[j, k] * w_r[k, i]
    return err, np.linalg.norm(err)


def _try_low_rank_search(n, m, p, max_rank, time_budget):
    """
    Attempt to find a rank < max_rank exact decomposition using iterative
    non-linear least squares optimization. Uses alternating least squares
    with structured initialization inspired by Strassen-like symmetries.
    
    Returns (U, V, W) as lists if a valid decomposition is found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n

    # Try ranks from a reasonable minimum up to max_rank-1
    # For 3x3x5, the theoretical minimum is unknown but schoolbook is 45
    # Let's try ranks from 20 to max_rank-1
    for rank in range(20, min(max_rank, 44)):
        if time.time() - _START_TIME > 150 - _DEADLINE_MARGIN:
            break

        # Multiple random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - _START_TIME > 150 - _DEADLINE_MARGIN:
                break

            np.random.seed(rank * 1000 + restart * 7 + 42)

            # Initialize with random factors
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5

            # Iterative refinement using gradient descent with adaptive step size
            max_iters = 500
            step_size = 0.1
            min_step = 1e-6
            patience = 50
            best_err = np.inf
            no_improve_count = 0

            for it in range(max_iters):
                if time.time() - _START_TIME > 150 - _DEADLINE_MARGIN:
                    break

                # Compute error and gradients
                err, err_norm = _compute_error(U, V, W, n, m, p)

                if err_norm < 1e-10:
                    # Found a valid decomposition!
                    # Round to nearest half-integer and verify
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2

                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()

                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Try without rounding
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        break

                # Track best error
                if err_norm < best_err:
                    best_err = err_norm
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        # Reduce step size
                        step_size *= 0.5
                        if step_size < min_step:
                            break
                        no_improve_count = 0

                # Compute gradients for U, V, W
                # dE/dU[i*m+j, r] = -2 * sum_k err[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                # dE/dV[j*p+k, r] = -2 * sum_i err[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                # dE/dW[k*n+i, r] = -2 * sum_j err[i,j,k] * U[i*m+j, r] * V[j*p+k, r]

                dU = np.zeros((nm, rank))
                dV = np.zeros((mp, rank))
                dW = np.zeros((pn, rank))

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = err[i, j, k]
                            for r in range(rank):
                                dU[i * m + j, r] -= 2 * e * V[j * p + k, r] * W[k * n + i, r]
                                dV[j * p + k, r] -= 2 * e * U[i * m + j, r] * W[k * n + i, r]
                                dW[k * n + i, r] -= 2 * e * U[i * m + j, r] * V[j * p + k, r]

                # Gradient step
                U -= step_size * dU
                V -= step_size * dV
                W -= step_size * dW

    return None


def construct():
    """
    Entry point with deadline guard. Returns the best valid decomposition found,
    falling back to the schoolbook rank-45 decomposition if no better result is found.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > 150 - _DEADLINE_MARGIN:
        # Fallback: return schoolbook decomposition (valid but rank 45)
        return _schoolbook_decomposition()

    n, m, p = DIMENSIONS

    # First, try to find a lower-rank decomposition
    time_budget = 150 - _DEADLINE_MARGIN - (time.time() - _START_TIME)
    if time_budget > 5:
        result = _try_low_rank_search(n, m, p, 45, time_budget)
        if result is not None:
            return result

    # Fall back to schoolbook decomposition
    return _schoolbook_decomposition()
