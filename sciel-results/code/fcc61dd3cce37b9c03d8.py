import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2

def _residuals(params, n, m, p, r):
    """Compute residuals for the decomposition."""
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = params[n*m*r+m*p*r:].reshape(p*n, r)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                residuals.append(val - 1.0)
    return np.array(residuals)

def _optimize_rank(n, m, p, r, time_limit=5.0, num_restarts=10):
    """
    Try to find a decomposition of rank r using optimization.
    """
    start_time = time.time()
    total_vars = (n*m + m*p + p*n) * r
    
    best_solution = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with small random values
        np.random.seed(random.randint(0, 2**31 - 1))
        params_init = np.random.randn(total_vars) * 0.1
        
        try:
            result = least_squares(
                _residuals, params_init,
                args=(n, m, p, r),
                method='lm',
                max_nfev=1000
            )
        except Exception:
            continue
        
        error = np.sum(result.fun**2)
        if error < best_error:
            best_error = error
            best_solution = result.x
        
        # Check if we found an exact solution
        if error < 1e-12:
            break
    
    if best_solution is None:
        return None
    
    # Extract U, V, W
    U = best_solution[:n*m*r].reshape(n*m, r)
    V = best_solution[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = best_solution[n*m*r+m*p*r:].reshape(p*n, r)
    
    # Round to half-integers
    U_rounded = _round_to_half_integer(U)
    V_rounded = _round_to_half_integer(V)
    W_rounded = _round_to_half_integer(W)
    
    # Verify the rounded solution
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # If rounding didn't work, try the unrounded solution (shouldn't happen per problem spec)
    # But return None if it's not exact
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Set a deadline for the whole construct function
    total_deadline = time.time() + 120  # 120s limit
    
    # Generate schoolbook solution first to have a fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to improve: search from rank 59 down to 48
    for r in range(rank_schoolbook - 1, max(47, 0), -1):
        if time.time() > total_deadline - 5:
            break
        
        # Allocate time for this rank
        time_left = total_deadline - time.time() - 5
        if time_left < 1:
            break
        
        alloc_time = min(5.0, time_left)
        
        result = _optimize_rank(n, m, p, r, alloc_time, num_restarts=5)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Return schoolbook
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }
