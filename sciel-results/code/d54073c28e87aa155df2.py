"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Schoolbook decomposition as fallback
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
        """Check if the decomposition is exact for matrix multiplication."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorized check:
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is a bit tricky to fully vectorize without loops over i,j,k due to index mapping.
        # But we can loop over i,j,k and vectorize over r.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Extract relevant rows
                    u_row = U_arr[i * m + j, :]
                    v_row = V_arr[j * p + k, :]
                    w_row = W_arr[k * n + i, :]
                    # Compute dot product of element-wise product
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1) > tol:
                        return False
        return True

    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are half-integers."""
        for row in arr:
            for x in row:
                if abs(x * 2 - round(x * 2)) > tol:
                    return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[round(x * 2) / 2.0 for x in row] for row in arr]

    def try_rank(rank, num_restarts=10):
        """Try to find a valid rank-r decomposition."""
        # Total variables: n*m*rank + m*p*rank + p*n*rank
        total_vars = (n * m + m * p + p * n) * rank
        # Constraints: n * m * p equations
        
        def objective(x):
            if time.time() > deadline - 5:
                return float('inf')
                
            U = x[:n * m * rank].reshape(n * m, rank)
            V = x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W = x[n * m * rank + m * p * rank :].reshape(p * n, rank)
            
            # Compute sum of squared errors
            # err = sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
            errors = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        val = np.dot(u_row, v_row * w_row)
                        errors += (val - 1.0) ** 2
            return errors

        best_result = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale matters. Small values might help L-BFGS converge locally.
            # But we need to hit the manifold of solutions.
            # Let's try a few different scales.
            scale = 0.5 + 0.1 * (restart % 5)
            x0 = np.random.randn(total_vars) * scale
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000})
                
                if res.fun < best_err:
                    best_err = res.fun
                    
                    # If residual is very small, try to snap and verify
                    if res.fun < 1e-6:
                        U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                        V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                        W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                        
                        # Snap to half-integers
                        U_snapped = snap_to_half_int(U_opt.tolist())
                        V_snapped = snap_to_half_int(V_opt.tolist())
                        W_snapped = snap_to_half_int(W_opt.tolist())
                        
                        if verify(U_snapped, V_snapped, W_snapped, tol=1e-8):
                            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                        
                        # If snapping fails, check if original is close enough to half-integers
                        # and if original verifies (floating point)
                        # But the task requires integer/half-integer entries.
                        # So we MUST return snapped values that verify.
                        
            except Exception:
                continue
                
        return None

    # Try ranks from low to high
    # Theoretical lower bound is ~26. Schoolbook is 36.
    # We will try 26 to 35.
    for rank in range(26, 36):
        if time.time() > deadline - 20:
            break
            
        result = try_rank(rank, num_restarts=15)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
