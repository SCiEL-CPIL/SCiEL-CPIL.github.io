import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120.0 - 15.0  # 105s safety margin

    n, m, p = DIMENSIONS

    # Build the target tensor T with shape (n, m, m, p, p, n)
    # T[i, j, j2, k, k2, i2] = 1 if j==j2 and i==i2 and k==k2 else 0
    T = np.zeros((n, m, m, p, p, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0

    T_flat = T.flatten()

    def compute_residual(U, V, W):
        """Compute residual for given factor matrices (shapes (n*m, R), (m*p, R), (p*n, R))"""
        R = U.shape[1]
        U_r = U.reshape(R, n, m)  # (R, n, m)
        V_r = V.reshape(R, m, p)  # (R, m, p)
        W_r = W.reshape(R, p, n)  # (R, p, n)

        recon = np.zeros_like(T)
        for r in range(R):
            u = U_r[r]  # (n, m)
            v = V_r[r]  # (m, p)
            w = W_r[r]  # (p, n)
            # outer product of u and v gives (n, m, m, p)
            uv = np.einsum('ij,jk->ijjk', u, v)
            # multiply by w: w is (p, n), we need to contract k with the 4th index and i with the 6th
            # recon[i,j,j2,k,k2,i2] += uv[i,j,j2,k] * w[k2,i2]
            recon += np.einsum('ijjk,ab->ijkab', uv, w)

        return recon.flatten() - T_flat

    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        R = U.shape[1]
        residual = compute_residual(U, V, W)
        return np.max(np.abs(residual)) < tol

    def snap_to_half_integers(arr, tol=1e-4):
        """Try to snap values to half-integers"""
        snapped = np.round(arr * 2) / 2
        # Only return if close enough
        if np.max(np.abs(snapped - arr)) < tol:
            return snapped
        return None

    def get_schoolbook_solution():
        """Return the standard schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

        return U, V, W

    def get_strassen_winslow_solution():
        """
        Attempt to construct a solution using block decomposition.
        For 3x4 * 4x7, we can split A into [A1 (2x4); A2 (1x4)] and B into [B1 (4x2) B2 (4x5)]?
        Or better, use the structure of small matmul tensors.
        
        Actually, let's try a specific heuristic:
        The rank of 3x4x7 is unknown but likely close to schoolbook or slightly less.
        Known lower bounds: max(3*4, 4*7, 3*7) = 28.
        Known upper bound: 84.
        
        Let's try to find a solution with rank < 84 using a structured initialization.
        We will initialize the factors with a "block" structure that mimics Strassen-like
        cancellations, then optimize.
        """
        pass

    def try_optimize(R, n_restarts=2, warm_start=None):
        """Try to find a solution of rank R"""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        best_result = None
        best_cost = np.inf

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if warm_start is not None and restart == 0:
                x0 = warm_start
            else:
                # Random initialization, scaled
                x0 = np.random.randn(total_size) * 0.5

            try:
                def residual_fn(params):
                    U_flat = params[:size_U]
                    V_flat = params[size_U:size_U + size_V]
                    W_flat = params[size_U + size_V:]

                    U = U_flat.reshape((n * m, R))
                    V = V_flat.reshape((m * p, R))
                    W = W_flat.reshape((p * n, R))

                    return compute_residual(U, V, W)

                res = least_squares(
                    residual_fn, x0,
                    method='lm', max_nfev=300,
                    xtol=1e-10, ftol=1e-10, gtol=1e-10
                )

                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res

                    if best_cost < 1e-14:
                        break

            except Exception:
                continue

        if best_result is None:
            return None

        if best_result.cost < 1e-12:
            size_U = n * m * R
            size_V = m * p * R

            U_flat = best_result.x[:size_U]
            V_flat = best_result.x[size_U:size_U + size_V]
            W_flat = best_result.x[size_U + size_V:]

            U = U_flat.reshape((n * m, R))
            V = V_flat.reshape((m * p, R))
            W = W_flat.reshape((p * n, R))

            # Try snapping to half-integers
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)

            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap

            if verify_solution(U, V, W):
                return U, V, W

        return None

    # Find the minimum rank
    best_solution = None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # But known results suggest we can do much better
    # For 3x4x7, try from 55 up
    lower_bound = 55
    upper_bound = n * m * p  # 84

    for R in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break

        result = try_optimize(R, n_restarts=2)

        if result is not None:
            U, V, W = result
            best_solution = (U, V, W)
            break

    # If no better solution found, use schoolbook
    if best_solution is None:
        U, V, W = get_schoolbook_solution()
    else:
        U, V, W = best_solution

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
