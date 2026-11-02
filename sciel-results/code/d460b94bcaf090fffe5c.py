import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def round_to_half(x):
        """Round array to nearest multiple of 0.5"""
        return np.round(x * 2) / 2.0

    def make_objective(rank, lambda_reg=1e-4):
        """Create the objective function for given rank."""
        def objective(x):
            # Reshape the flat vector into U, V, W
            size_u = n * m * rank
            size_v = m * p * rank
            size_w = p * n * rank
            
            U = x[:size_u].reshape(n, m, rank)
            V = x[size_u:size_u + size_v].reshape(m, p, rank)
            W = x[size_u + size_v:].reshape(p, n, rank)
            
            # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            
            # Data term: (T - 1)^2
            data_term = np.sum((T - 1.0) ** 2)
            
            # Regularization term: keep values small to avoid overflow and help convergence
            reg_term = lambda_reg * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            
            return data_term + reg_term
        
        return objective

    def try_rank(rank, time_limit=20.0):
        """Try to find a decomposition of the given rank within the time limit."""
        num_vars = (n * m + m * p + p * n) * rank
        obj_func = make_objective(rank)
        
        # Estimate max trials
        max_trials = int(time_limit / 0.2)
        max_trials = max(10, min(max_trials, 200)) # Cap between 10 and 200
        
        current_time = time.time()
        trial_start = current_time
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0 or (time.time() - trial_start > time_limit):
                break
            
            # Random initialization
            # Scale to 1/sqrt(rank) to keep initial magnitudes moderate
            scale = 1.0 / np.sqrt(rank)
            x0 = np.random.randn(num_vars) * scale
            
            try:
                res = minimize(
                    obj_func, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            except Exception:
                continue
            
            # Check result
            if res.fun < 1e-6: # Relaxed threshold for initial check
                x = res.x
                size_u = n * m * rank
                size_v = m * p * rank
                size_w = p * n * rank
                
                U = x[:size_u].reshape(n, m, rank)
                V = x[size_u:size_u + size_v].reshape(m, p, rank)
                W = x[size_u + size_v:].reshape(p, n, rank)
                
                # Attempt to round to half-integers
                U_rounded = round_to_half(U)
                V_rounded = round_to_half(V)
                W_rounded = round_to_half(W)
                
                # Verify exactness with rounded values
                T_rounded = np.einsum('ijr,jkr,kir->ijk', U_rounded, V_rounded, W_rounded)
                err_rounded = np.max(np.abs(T_rounded - 1.0))
                
                if err_rounded < 1e-9:
                    return {
                        "U": U_rounded.reshape(n * m, rank).tolist(),
                        "V": V_rounded.reshape(m * p, rank).tolist(),
                        "W": W_rounded.reshape(p * n, rank).tolist()
                    }
                
                # If rounding fails, check if the float solution is extremely precise
                # and maybe the evaluator is lenient? But task says "integer/half-integer".
                # So we strictly require the rounded version to work.
                # If not, we might try to polish the float solution further?
                # For now, skip this trial if rounding doesn't yield a valid solution.
        
        return None

    # Try ranks from 19 upwards
    # Rank 19 is the theoretical minimum for 2x3x4 matrix multiplication
    # Allocate time based on likelihood
    time_allocs = {
        19: 50.0,
        20: 30.0,
        21: 15.0,
        22: 5.0
    }
    
    for r in range(19, 23):
        if time.time() > deadline - 10.0:
            break
        
        sol = try_rank(r, time_limit=time_allocs.get(r, 5.0))
        if sol is not None:
            return sol
    
    # Fallback to schoolbook decomposition with rank = n*m*p = 24
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
