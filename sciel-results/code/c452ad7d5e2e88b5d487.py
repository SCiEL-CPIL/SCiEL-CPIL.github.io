import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    rank_schoolbook = n * m * p  # 42

    # Build target tensor T
    # T[alpha, beta, gamma] = 1 if indices match
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    T_flat = T.flatten()
    N_total = N_U + N_V + N_W

    def make_residual_func(R):
        def res(params):
            # Unpack
            len_U = N_U * R
            len_V = N_V * R
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            
            # Compute Tensor Sum
            # T_calc[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            return (T_calc.flatten() - T_flat)
        
        return res

    def try_rank(R, time_limit):
        if R <= 0:
            return None
        
        res_func = make_residual_func(R)
        total_vars = N_total * R
        
        best_sol = None
        best_cost = np.inf
        
        start_time_rank = time.time()
        
        # Number of restarts
        # We can run as many as we can within time_limit
        # Each restart is very fast (<10ms)
        # So 100 restarts is safe.
        num_restarts = 100
        
        for restart in range(num_restarts):
            if time.time() > start_time_rank + time_limit:
                break
            
            # Random initialization in [-5, 5] to allow larger coefficients
            x0 = np.random.uniform(-5, 5, total_vars)
            
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # Check if error is close to 0
        # If cost < 1e-6, try rounding
        if best_cost < 1e-4:
            # Round to half-integers
            # Multiply by 2, round, divide by 2
            x_rounded = np.round(best_sol * 2) / 2.0
            
            # Verify exactness with rounded solution
            res_rounded = res_func(x_rounded)
            if np.linalg.norm(res_rounded) < 1e-9:
                # Extract U, V, W
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded[:len_U].reshape(N_U, R)
                V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded[len_U+len_V:].reshape(N_W, R)
                
                # Convert to list of lists for output
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # First, ensure we have the schoolbook solution as a fallback
    U = [[0] * rank_schoolbook for _ in range(N_U)]
    V = [[0] * rank_schoolbook for _ in range(N_V)]
    W = [[0] * rank_schoolbook for _ in range(N_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # Iterate from rank_schoolbook - 1 down to 20
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time: 2 seconds per rank is safe and allows ~70 ranks if needed
        # But we only have ~21 ranks.
        time_per_rank = 2.0
        
        result = try_rank(R, time_per_rank)
        if result is not None:
            best_result = result
            best_rank = R
            # Continue searching for lower ranks
            
    return best_result
# EVOLVE-BLOCK-END
