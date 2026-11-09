import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    def _schoolbook():
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
        return U, V, W, rank

    def _verify(U, V, W, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Schoolbook is the fallback
    U, V, W, best_rank = _schoolbook()
    
    # Try to find lower rank decompositions using optimization
    # Target ranks to try
    target_ranks = [46, 47, 48, 49]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random values
            np.random.seed(restart)
            U_flat = np.random.randn(n * m * target_rank) * 0.1
            V_flat = np.random.randn(m * p * target_rank) * 0.1
            W_flat = np.random.randn(p * n * target_rank) * 0.1
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                U = U_flat.reshape(n * m, target_rank)
                V = V_flat.reshape(m * p, target_rank)
                W = W_flat.reshape(p * n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k]) * 1.0 # Wait, W is also needed
                            # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # This is a tensor contraction.
                            # Efficiently:
                            # For each r, we have U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                            # We can compute this using einsum or loops.
                            # Given small sizes, loops are fine.
                            pass
                            
                # Better objective calculation
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # We can vectorize this:
                            # u = U_mat[i*m+j]
                            # v = V_mat[j*p+k]
                            # w = W_mat[k*n+i]
                            # s = np.einsum('r,r,r->', u, v, w)
                            s = np.einsum('r,r,r->', U_mat[i*m+j], V_mat[j*p+k], W_mat[k*n+i])
                            err += (s - 1.0) ** 2
                return err

            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
                if res.fun < 1e-4:
                    # Try to round to half-integers
                    U_opt = res.x[:n*m*target_rank].reshape(n * m, target_rank)
                    V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p * n, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if _verify(U_list, V_list, W_list):
                        U, V, W, best_rank = U_list, V_list, W_list, target_rank
                        found = True
                        break
            except Exception:
                pass
                
        if found:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
