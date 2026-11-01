import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 32)
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Strategy: Use continuous optimization (L-BFGS-B) to find a lower-rank decomposition.
    # We parameterize U, V, W as continuous matrices and minimize the squared error.
    # If we find a solution with error ~0, we round to integers and verify.
    
    def flatten_params(params, r):
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        U = params[:u_size].reshape(n * m, r)
        V = params[u_size:u_size+v_size].reshape(m * p, r)
        W = params[u_size+v_size:].reshape(p * n, r)
        return U, V, W
    
    def objective(params, r):
        U, V, W = flatten_params(params, r)
        # Compute tensor T_sum
        # T_sum[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Using einsum for efficiency
        T_sum = np.einsum('ar,br,cr->abc', U, V, W)
        # Minimize ||T_sum - T||^2
        diff = T_sum - T
        return np.sum(diff * diff)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try ranks from 31 down to 1
    # We will spend a limited amount of time on each rank.
    # Since 2x2x8 is not a standard square case, the rank might be close to 32 or significantly lower.
    # Let's try a few specific ranks: 31, 30, 28, 24, 20, 16, 12, 8, 4, 2, 1
    # Or just iterate down.
    
    # To speed up, we can use random restarts for each rank.
    # We'll use a small number of restarts per rank, with a time limit per rank.
    
    ranks_to_try = list(range(best_rank - 1, 0, -1))
    
    # Prioritize ranks that are likely candidates.
    # For 2x2x2, rank is 7. For 2x2x8, if it were a direct sum, it would be 7*8=56 (worse).
    # But it's a single tensor.
    # Let's just try all ranks down to 1, but with a strict time budget per rank.
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Time budget for this rank
        # We want to try a few restarts.
        # Let's say 2 seconds per rank on average? No, that's too short for optimization.
        # Let's say we have ~140s total. 31 ranks. ~4.5s per rank.
        # We'll do 3 restarts per rank, with a time limit of 1.5s per restart.
        
        max_restarts = 3
        time_per_restart = 1.5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize parameters
            # Random initialization
            u_size = n * m * r
            v_size = m * p * r
            w_size = p * n * r
            total_size = u_size + v_size + w_size
            
            # Use small random values
            params0 = np.random.randn(total_size) * 0.1
            
            # Optimize
            res = minimize(objective, params0, args=(r,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
            
            # Check if we found a good solution
            if res.fun < 1e-6:
                U, V, W = flatten_params(res.x, r)
                
                # Round to nearest integer
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                
                # Verify
                T_sum = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.array_equal(T_sum, T):
                    best_rank = r
                    best_result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    break # Move to next rank
        
    return best_result
# EVOLVE-BLOCK-END
