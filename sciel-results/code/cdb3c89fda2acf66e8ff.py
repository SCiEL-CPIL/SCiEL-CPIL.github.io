import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build the tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose T exactly."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        error = np.max(np.abs(T - T_recon))
        return error < tol, error

    def try_rank_exact(r, max_attempts=20):
        """Try to find exact rank-r decomposition using algebraic approach."""
        if time.time() > deadline:
            return None
        
        # Reshape T for least squares
        T_w = T.reshape(n*m * m*p, p*n)
        T_v = np.transpose(T, (0, 2, 1)).reshape(n*m * p*n, m*p)
        T_u = np.transpose(T, (1, 2, 0)).reshape(m*p * p*n, n*m)

        best_result = None
        best_error = float('inf')

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Initialize with structured random
            np.random.seed(42 + attempt * 137)
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            elif attempt % 3 == 1:
                # Sparse initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for d in range(r):
                    # Each rank-1 component covers some subset
                    idx_u = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                    idx_v = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                    idx_w = np.random.choice(p*n, size=max(1, p*n//4), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            else:
                # Diagonal-like initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for d in range(r):
                    U[d % (n*m), d] = 1
                    V[d % (m*p), d] = 1
                    W[d % (p*n), d] = 1

            # Alternating least squares with multiple inner iterations
            for it in range(150):
                if time.time() > deadline:
                    break

                # Solve for W given U, V
                A_w = np.einsum('ad,bd->abd', U, V).reshape(n*m * m*p, r)
                W_sol, residuals, rank_w, sv = np.linalg.lstsq(A_w, T_w, rcond=None)
                W = W_sol.T

                # Solve for V given U, W
                A_v = np.einsum('ad,cd->acd', U, W).reshape(n*m * p*n, r)
                V_sol, residuals, rank_v, sv = np.linalg.lstsq(A_v, T_v, rcond=None)
                V = V_sol.T

                # Solve for U given V, W
                A_u = np.einsum('bd,cd->bcd', V, W).reshape(m*p * p*n, r)
                U_sol, residuals, rank_u, sv = np.linalg.lstsq(A_u, T_u, rcond=None)
                U = U_sol.T

                # Check convergence
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                error = np.linalg.norm(T - T_recon)
                
                if error < best_error:
                    best_error = error
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if error < 1e-10:
                    break

            if best_error < 1e-8:
                # Try to round to half-integers
                U_r = np.round(best_result[0] * 2) / 2
                V_r = np.round(best_result[1] * 2) / 2
                W_r = np.round(best_result[2] * 2) / 2
                
                ok, err_r = verify(U_r, V_r, W_r)
                if ok:
                    return U_r, V_r, W_r
                
                # Try integer rounding
                U_i = np.round(best_result[0])
                V_i = np.round(best_result[1])
                W_i = np.round(best_result[2])
                ok, err_i = verify(U_i, V_i, W_i)
                if ok:
                    return U_i, V_i, W_i

                # If not exactly roundable but very close, check if it's truly exact
                ok, err = verify(best_result[0], best_result[1], best_result[2])
                if ok and err < 1e-10:
                    # Return as-is (should be very close to rational)
                    return best_result[0], best_result[1], best_result[2]

        return None

    # Try ranks from 14 upward
    for r in [14, 15]:
        if time.time() > deadline:
            break
        
        result = try_rank_exact(r)
        if result is not None:
            U, V, W = result
            # Final verification
            ok, err = verify(U, V, W)
            if ok:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Try rank 16 (schoolbook) as guaranteed fallback
    return schoolbook()
