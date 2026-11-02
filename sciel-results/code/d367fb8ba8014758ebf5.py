import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin

    n, m, p = DIMENSIONS

    # Target tensor T_flat
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # We can avoid storing T explicitly if we compute the difference on the fly, 
    # but T is small enough (14112 floats).
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    T_flat = T.flatten()

    def compute_residual(params, R):
        """Compute residual for given parameters and rank, vectorized."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R

        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]

        # Reshape to (R, n, m), (R, m, p), (R, p, n)
        U = U_flat.reshape((R, n, m))
        V = V_flat.reshape((R, m, p))
        W = W_flat.reshape((R, p, n))

        # Compute reconstruction
        # uv[i, j, j2, k] = sum_r U[r, i, j] * V[r, j2, k]
        uv = np.einsum('rij,rjk->rijjk', U, V)
        # recon[i, j, j2, k, i2, k2] = sum_r uv[r, i, j, j2, k] * W[r, k2, i2]
        recon = np.einsum('rijjk,rki->ijkiki', uv, W)

        return (recon.flatten() - T_flat)

    def try_rank(R, n_restarts=10, max_nfev=200):
        """Try to find a valid decomposition at rank R."""
        total_dim = n * m * R + m * p * R + p * n * R

        for attempt in range(n_restarts):
            if time.time() > deadline:
                return None

            # Initialization
            # Try a mix of small random and structured
            if attempt % 2 == 0:
                x0 = np.random.randn(total_dim) * 0.5
            else:
                # Initialize with a random rank-1 approximation scaled
                # Or just random
                x0 = np.random.randn(total_dim) * 0.1

            try:
                res = least_squares(
                    compute_residual, x0, args=(R,),
                    method='lm', max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
            except Exception:
                continue

            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R

                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]

                # Try snapping to nearest integer
                U_int = np.round(U_flat)
                V_int = np.round(V_flat)
                W_int = np.round(W_flat)
                
                x_snapped = np.concatenate([U_int, V_int, W_int])
                res_snapped = compute_residual(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    return {
                        "U": U_int.reshape((n * m, R)).tolist(),
                        "V": V_int.reshape((m * p, R)).tolist(),
                        "W": W_int.reshape((p * n, R)).tolist()
                    }

                # Try snapping to nearest half-integer
                U_half = np.round(U_flat * 2) / 2
                V_half = np.round(V_flat * 2) / 2
                W_half = np.round(W_flat * 2) / 2
                
                x_snapped = np.concatenate([U_half, V_half, W_half])
                res_snapped = compute_residual(x_snapped, R)

                if np.max(np.abs(res_snapped)) < 1e-9:
                    return {
                        "U": U_half.reshape((n * m, R)).tolist(),
                        "V": V_half.reshape((m * p, R)).tolist(),
                        "W": W_half.reshape((p * n, R)).tolist()
                    }
                    
                # If snapping fails, return the float solution? 
                # The task asks for integer/half-integer entries.
                # If we can't snap, this rank might not be valid for the "exact" requirement 
                # or we need a better local search.
                # However, if the residual is extremely small, it might be a numerical artifact.
                # Let's try a few more iterations or accept if it's very close?
                # The prompt says "exactly decompose". So we must snap.
                
        return None

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 20 to 84
    # We can parallelize or just loop.
    for R in range(20, min(85, n * m * p)):
        if time.time() > deadline:
            break

        # Use more restarts for lower ranks
        n_restarts = 15 if R <= 40 else 10
        max_nfev = 300 if R <= 40 else 200

        sol = try_rank(R, n_restarts=n_restarts, max_nfev=max_nfev)
        if sol is not None:
            best_rank = R
            best_solution = sol
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook decomposition
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
