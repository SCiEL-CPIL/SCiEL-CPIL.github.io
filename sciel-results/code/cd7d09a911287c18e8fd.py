# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    K_values = [50, 100, 150, 200, 300, 500]
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = -1e18
    
    for K in K_values:
        if time.time() > deadline - 180:
            break
            
        n_vars = K - 1
        if n_vars < 1:
            continue
            
        k_range = np.arange(2, K + 1)
        c = np.log(k_range) / k_range
        
        N = 10 * K
        x_range = np.arange(1, N + 1)
        
        A_ub = np.zeros((N, n_vars), dtype=np.float64)
        
        for j, k in enumerate(k_range):
            floor_xk = x_range // k
            xk = x_range.astype(np.float64) / k
            A_ub[:, j] = floor_xk.astype(np.float64) - xk
        
        b_ub = np.ones(N, dtype=np.float64) * 1.0001
        bounds = [(-10.0, 10.0)] * n_vars
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                f_vals = result.x
                score = -np.sum(f_vals * c)
                
                M_vals = A_ub @ f_vals
                max_M = np.max(M_vals)
                
                if max_M <= 1.0001 + 1e-9:
                    if score > best_score:
                        best_score = score
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[1:] = f_vals
                        best_domain = domain
                        best_values = values
                        
        except Exception as e:
            continue
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
