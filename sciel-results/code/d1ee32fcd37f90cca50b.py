import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    # Precompute the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def objective(x, rank):
        """Flattened objective function for scipy.optimize.minimize."""
        U = x[:n * m * rank].reshape(n * m, rank)
        V = x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
        W = x[n * m * rank + m * p * rank :].reshape(p * n, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = T - T_target
        return np.sum(err ** 2)

    def try_rank(rank, max_iterations=100, seed=42):
        rng = np.random.RandomState(seed)
        best_x = None
        best_err = float('inf')
        
        for restart in range(max_iterations):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            x0 = rng.randn(n * m * rank + m * p * rank + p * n * rank) * 0.1
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    objective, 
                    x0, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
                    
                    if best_err < 1e-10:
                        # Found a solution
                        U = best_x[:n * m * rank].reshape(n * m, rank)
                        V = best_x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                        W = best_x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                        
                        # Round to nearest 0.5
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        # Verify rounded
                        T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        if np.max(np.abs(T_r - T_target)) < 1e-10:
                            return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                continue

        # If we found a good approximate solution, try to round it
        if best_x is not None and best_err < 1e-6:
            U = best_x[:n * m * rank].reshape(n * m, rank)
            V = best_x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W = best_x[n * m * rank + m * p * rank :].reshape(p * n, rank)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.max(np.abs(T_r - T_target)) < 1e-10:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None

    # Fallback schoolbook
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Try ranks from 23 upwards
    for rank in range(23, 35):
        if time.time() > deadline - 20:
            break
        res = try_rank(rank, max_iterations=5, seed=rank)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
