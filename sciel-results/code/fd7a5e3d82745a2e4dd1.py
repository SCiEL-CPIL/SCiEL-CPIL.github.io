import numpy as np
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    best_score = 0.0
    best_domain = np.array([1], dtype=np.int64)
    best_values = np.array([0.0], dtype=np.float64)
    
    # We can try larger K now. HiGHS is very fast for small problems.
    # K=50 means 49 variables and 500 constraints.
    max_K = 50
    
    for K in range(2, max_K + 1):
        n_vars = K - 1
        if n_vars == 0:
            continue
            
        # Objective: minimize sum v_k ln(k)/k
        c = np.zeros(n_vars)
        for i in range(n_vars):
            k = i + 2
            c[i] = np.log(k) / k
            
        # Constraints: M(x) <= 1 for x in 1..10K
        # M(x) = sum v_k (floor(x/k) - x/k)
        A_ub = np.zeros((10 * K, n_vars))
        b_ub = np.ones(10 * K)
        
        # Precompute logs and inverses? Not strictly necessary for small K.
        for x in range(1, 10 * K + 1):
            for i in range(n_vars):
                k = i + 2
                A_ub[x-1, i] = np.floor(x / k) - x / k
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                v = res.x
                # Calculate score a = - sum v_k ln(k)/k
                score = -np.dot(v, c)
                if score > best_score:
                    best_score = score
                    domain = np.array([1] + list(range(2, K + 1)), dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = 1.0
                    values[1:] = v
                    best_domain = domain
                    best_values = values
        except Exception:
            pass
            
    # Fallback
    if best_score < 0.6:
        best_domain = np.array([1, 2], dtype=np.int64)
        best_values = np.array([1.0, -2.0], dtype=np.float64)
        best_score = np.log(2)
        
    return best_domain, best_values
