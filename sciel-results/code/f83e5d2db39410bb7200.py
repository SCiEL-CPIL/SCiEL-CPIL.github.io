import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    indices = [(i, j, k) for i in range(n) for j in range(m) for j in range(m) for k in range(p)] # Correct loop
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    u_idx = np.array([i * m + j for (i, j, k) in indices])
    v_idx = np.array([j * p + k for (i, j, k) in indices])
    w_idx = np.array([k * n + i for (i, j, k) in indices])

    def residual(params, rank):
        U = params[:nm * rank].reshape(nm, rank)
        V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = params[nm * rank + mp * rank:].reshape(pn, rank)
        prod = V[v_idx, :] * W[w_idx, :]
        res = np.einsum('ir,ir->i', U[u_idx, :], prod) - 1.0
        return res

    def verify(U, V, W):
        for (i, j, k) in indices:
            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def snap_and_verify(arr, rank):
        # Try to snap to half-integers
        r = np.round(arr * 2) / 2.0
        if np.max(np.abs(arr - r)) < 1e-6:
            return r
        return arr

    best_rank = None
    best_solution = None

    # Try ranks from 18 to 23 (assuming 18 is possible, go up to 23)
    # If 18 is not found, try 19, etc.
    # Start from lower ranks to find the minimum.
    for rank in range(18, 24):
        if time.time() > deadline - 20:
            break

        # More attempts for lower ranks
        max_attempts = 100 if rank <= 19 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            np.random.seed(rank * 1000 + attempt * 13 + 7)
            # Scale initialization
            scale = 1.0
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                # Use LM method
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=1000)
                
                if result.cost < 1e-12:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)

                    # Verify
                    if verify(U_sol, V_sol, W_sol):
                        # Snap
                        U_s = snap_and_verify(U_sol, rank)
                        V_s = snap_and_verify(V_sol, rank)
                        W_s = snap_and_verify(W_sol, rank)
                        
                        if verify(U_s, V_s, W_s):
                            U_sol, V_sol, W_sol = U_s, V_s, W_s
                            
                        best_rank = rank
                        best_solution = (U_sol, V_sol, W_sol)
                        break
            except Exception:
                continue

        if best_rank is not None:
            break

    if best_rank is not None:
        U_sol, V_sol, W_sol = best_solution
        # Convert to list of lists
        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

    # Fallback: rank 24
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
