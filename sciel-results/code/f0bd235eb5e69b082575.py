import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 130  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    T = np.ones((n, m, p))
    
    def try_rank(R):
        num_params = R * (n * m + m * p + p * n)
        
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()

        # Try multiple random restarts
        for attempt in range(20):
            if time.time() - start_time > max_time:
                return None
            
            # Random initialization
            # Scale: entries are typically small integers.
            init_params = np.random.randn(num_params) * 0.5
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, init_params, method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                # If the solution is good, try to polish to integer/half-integer
                if sol.cost < 1e-6:
                    # Round to nearest 0.5
                    rounded_params = np.round(sol.x * 2) / 2.0
                    res_check = residuals(rounded_params)
                    
                    # Local search: check neighbors in the 0.5 lattice
                    # If the residual is not zero, try small perturbations?
                    # Or just check if the rounded version is exact.
                    if np.linalg.norm(res_check) < 1e-6:
                        U, V, W = unpack(rounded_params)
                        U3 = U.reshape(n, m, R)
                        V3 = V.reshape(m, p, R)
                        W3 = W.reshape(p, n, R)
                        A_exact = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                        if np.allclose(A_exact, T, atol=1e-9):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
                            
            except Exception:
                pass
                
        return None

    # Try ranks from 20 up to 23
    # It is highly likely that 20 is the minimum rank for 2x3x4.
    for R in range(20, target_rank):
        res = try_rank(R)
        if res:
            return res

    # Fallback: Schoolbook decomposition
    rank = target_rank
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
