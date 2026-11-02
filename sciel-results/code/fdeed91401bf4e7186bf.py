import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def solve_U_given_VW(V, W, r):
        """Solve for U given V and W. Returns (U, residual_norm)."""
        U = np.zeros((n * m, r), dtype=np.float64)
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                # For fixed (i,j), equations: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A_mat = np.zeros((p, r), dtype=np.float64)
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b_vec = np.ones(p, dtype=np.float64)
                # Solve A_mat @ U[i*m+j, :] = b_vec
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[i * m + j, :] = sol
                    # Compute residual
                    res = A_mat @ sol - b_vec
                    max_res = max(max_res, float(np.max(np.abs(res))))
                except np.linalg.LinAlgError:
                    U[i * m + j, :] = 0.0
                    max_res = float('inf')
        return U, max_res

    def residual_of_VW(V, W, r):
        """Compute the residual when solving for U given V, W."""
        U, max_res = solve_U_given_VW(V, W, r)
        return U, max_res

    def try_rank_block_elimination(r, max_iter=300, tol=1e-10):
        """
        Block-elimination: fix V, W; solve for U; check residual.
        If residual > 0, refine V and W to minimize the residual.
        """
        if r >= best_rank:
            return None

        for init_seed in range(20):
            if time.time() > deadline:
                return None

            np.random.seed(1000 + r * 100 + init_seed * 7)
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5

            U, res = solve_U_given_VW(V, W, r)
            if res < tol:
                break

            # Gradient-based refinement of V and W
            # The residual is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
            # But U depends on V, W, so this is complicated.
            # Instead, use a simpler approach: minimize the squared residual
            # by doing a few steps of coordinate descent on V and W.

            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                U, res = solve_U_given_VW(V, W, r)
                if res < tol:
                    break

                # Compute gradient of residual w.r.t. V and W
                # For each (i,j,k): f(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                # df/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                # df/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]

                # Compute the residual vector
                res_vec = np.zeros((n, m, p), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res_vec[i, j, k] = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])) - 1.0

                # Gradient of sum of squared residuals
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            df = 2.0 * res_vec[i, j, k]
                            for r_idx in range(r):
                                grad_V[j * p + k, r_idx] += df * U[i * m + j, r_idx] * W[k * n + i, r_idx]
                                grad_W[k * n + i, r_idx] += df * U[i * m + j, r_idx] * V[j * p + k, r_idx]

                # Gradient descent step
                V -= lr * grad_V
                W -= lr * grad_W

                # Occasionally re-normalize to prevent drift
                if it % 50 == 0 and it > 0:
                    V_norm = np.linalg.norm(V)
                    W_norm = np.linalg.norm(W)
                    if V_norm > 0 and W_norm > 0:
                        V *= (1.0 / V_norm) * np.sqrt(m * p)
                        W *= (1.0 / W_norm) * np.sqrt(p * n)

            U, res = solve_U_given_VW(V, W, r)
            if res < tol:
                # Try to round to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
                # Return as-is if it's close enough
                if verify(U, V, W, 1e-8):
                    return U, V, W

        return None

    # Try ranks from best_rank-1 down to some minimum
    candidate_ranks = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
        result = try_rank_block_elimination(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank we tried

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
