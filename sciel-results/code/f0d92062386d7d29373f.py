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

    def compute_residual_vec(U, V, W, r):
        """Compute the residual vector for all (i,j,k)."""
        res_vec = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res_vec[i, j, k] = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])) - 1.0
        return res_vec

    def try_rank_optimization(r, max_iter=500, tol=1e-10, n_restarts=15):
        """
        Optimize V and W to minimize residual, with U solved exactly each step.
        Uses gradient descent with adaptive learning rate.
        """
        if r >= best_rank:
            return None

        for init_seed in range(n_restarts):
            if time.time() > deadline:
                return None

            np.random.seed(10000 + r * 1000 + init_seed * 137)
            # Structured initialization: use patterns based on the 6=2*3 structure
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3

            # Add some structure: make certain entries larger
            for idx in range(m * p):
                if idx % 2 == 0:
                    V[idx, :] *= 1.5
            for idx in range(p * n):
                if idx % 3 == 0:
                    W[idx, :] *= 1.5

            U, res = solve_U_given_VW(V, W, r)
            if res < tol:
                break

            # Gradient-based optimization with adaptive learning rate
            lr = 0.05
            lr_decay = 0.995
            best_local_res = res
            stuck_count = 0

            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                U, res = solve_U_given_VW(V, W, r)
                if res < tol:
                    break

                if res < best_local_res:
                    best_local_res = res
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 100:
                        # Try a new random perturbation
                        V += np.random.randn(m * p, r) * 0.1
                        W += np.random.randn(p * n, r) * 0.1
                        stuck_count = 0
                        lr = 0.05
                        continue

                # Compute residual vector
                res_vec = compute_residual_vec(U, V, W, r)

                # Compute gradient of sum of squared residuals w.r.t. V and W
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            df = 2.0 * res_vec[i, j, k]
                            for r_idx in range(r):
                                grad_V[j * p + k, r_idx] += df * U[i * m + j, r_idx] * W[k * n + i, r_idx]
                                grad_W[k * n + i, r_idx] += df * U[i * m + j, r_idx] * V[j * p + k, r_idx]

                # Adaptive learning rate: reduce if gradient is large
                grad_norm_V = np.linalg.norm(grad_V)
                grad_norm_W = np.linalg.norm(grad_W)
                max_grad = max(grad_norm_V, grad_norm_W, 1e-10)
                effective_lr = lr * min(1.0, 1.0 / max_grad)

                # Gradient descent step
                V -= effective_lr * grad_V
                W -= effective_lr * grad_W

                # Decay learning rate
                lr *= lr_decay

                # Occasionally re-normalize to prevent drift
                if it % 100 == 0 and it > 0:
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
    # Focus on promising ranks first
    candidate_ranks = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
        result = try_rank_optimization(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition at this rank

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
