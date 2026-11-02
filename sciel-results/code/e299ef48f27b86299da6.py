import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 145
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank_val = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank_val):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute the sum of squared errors for the decomposition."""
        rank_val = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank_val):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1.0) ** 2
        return error

    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Try to reduce rank using numerical optimization
    # We try ranks from best_rank - 1 downwards
    
    # To speed up, we can use the schoolbook solution as a warm start
    # by setting the column to remove to zero and letting the optimizer fix it.
    
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 20:
            break
            
        # We try a few random restarts for this target rank
        found = False
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Initialize with schoolbook, zero out one random column
            U_init = U_school[:, :target_rank].copy()
            V_init = V_school[:, :target_rank].copy()
            W_init = W_school[:, :target_rank].copy()
            
            # If we just truncated, the error is high.
            # Better: Start with random small values for the last few columns?
            # Or use the schoolbook columns but allow them to change?
            
            # Let's use a random initialization for the last 10% of columns to help escape local minima
            if target_rank < rank_school:
                # Keep the first 90% as schoolbook, randomize the rest
                split = int(target_rank * 0.9)
                if split > 0:
                    U_init[:, :split] = U_school[:, :split]
                    V_init[:, :split] = V_school[:, :split]
                    W_init[:, :split] = W_school[:, :split]
                if split < target_rank:
                    U_init[:, split:] = np.random.uniform(-0.5, 0.5, (nm, target_rank - split))
                    V_init[:, split:] = np.random.uniform(-0.5, 0.5, (mp, target_rank - split))
                    W_init[:, split:] = np.random.uniform(-0.5, 0.5, (pn, target_rank - split))
            
            # Flatten parameters
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_curr = x[:nm * target_rank].reshape(nm, target_rank)
                V_curr = x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_curr = x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
                
                # Compute error
                err = 0.0
                # Optimization: Use vectorized operations if possible, but loops are okay for small N
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err += (val - 1.0) ** 2
                return err

            # Use L-BFGS-B for speed
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.success or res.fun < 1e-8:
                U_opt = res.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
                
                # Round to half-integers
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round, tol=1e-9):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    found = True
                    break # Found a valid decomposition for this rank, try lower
        
    # Final check
    if not verify(best_U, best_V, best_W, tol=1e-9):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
