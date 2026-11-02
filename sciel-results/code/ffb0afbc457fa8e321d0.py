import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120.0 - 15.0  # 105s safety margin

    n, m, p = DIMENSIONS

    # Build the target tensor T
    # T[i, j, j2, k, k2, i2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, p, n)
    T = np.zeros((n, m, m, p, p, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0

    T_flat = T.flatten()
    N_entries = len(T_flat)

    def compute_residual(U, V, W):
        """Compute residual for given factor matrices"""
        R = U.shape[1]
        U_r = U.reshape(R, n, m)
        V_r = V.reshape(R, m, p)
        W_r = W.reshape(R, p, n)

        recon = np.zeros((n, m, m, p, p, n), dtype=np.float64)
        for r in range(R):
            u = U_r[r]
            v = V_r[r]
            w = W_r[r]
            # u (n,m), v (m,p), w (p,n)
            # term[i,j,j2,k,k2,i2] = u[i,j] * v[j2,k] * w[k2,i2]
            # We need to sum over r.
            # Using einsum:
            # u[i,j] v[j2,k] w[k2,i2] -> i j j2 k k2 i2
            recon += np.einsum('ij,j2k,k2i2->ijj2kk2i2', u, v, w)
        
        return recon.flatten() - T_flat

    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        residual = compute_residual(U, V, W)
        return np.max(np.abs(residual)) < tol

    def snap_to_half_integers(arr, tol=1e-4):
        """Try to snap values to half-integers"""
        snapped = np.round(arr * 2) / 2
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

    def try_optimize(R, n_restarts=5):
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

            x0 = np.random.randn(total_size) * 0.1

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
                    method='lm', max_nfev=500,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )

                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res

                    if best_cost < 1e-16:
                        break

            except Exception:
                continue

        if best_result is None:
            return None

        if best_result.cost < 1e-10:
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
                # If it verifies but didn't snap, we might still return it if the problem allows reals?
                # Contract says "integer/half-integer entries". 
                # If it doesn't snap, it's invalid for the contract?
                # "factor matrices ... with integer/half-integer entries that exactly decompose"
                # So we MUST snap. If snap fails, we reject this solution.
                return None

        return None

    # Find the minimum rank
    best_solution = None
    schoolbook_U, schoolbook_V, schoolbook_W = get_schoolbook_solution()
    
    # Try ranks from 83 down to a lower bound
    # If we find one, we return it.
    # If we don't, we return schoolbook (84).
    
    # To save time, we can try a few random restarts for rank 83 first.
    # If that fails, we might assume 84 is the min or close to it.
    
    # Let's try 83, 82, 81...
    for R in range(83, 50, -1):
        if time.time() > deadline:
            break
        
        result = try_optimize(R, n_restarts=3)
        if result is not None:
            U, V, W = result
            best_solution = (U, V, W)
            break

    if best_solution is None:
        U, V, W = schoolbook_U, schoolbook_V, schoolbook_W
    else:
        U, V, W = best_solution

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
