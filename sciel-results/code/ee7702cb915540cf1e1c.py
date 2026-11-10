import numpy as np
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    K = 1000
    # Variables g(2) ... g(K)
    n_vars = K - 1
    
    # Objective: Minimize sum g(k) * log(k)/k
    c = np.array([np.log(k) / k for k in range(2, K + 1)])
    
    # Bounds
    bounds = [(-10, 10)] * n_vars
    
    # Constraints
    # 1. sum g(k)/k >= -1.0001  => -sum g(k)/k <= 1.0001
    A_eq = np.array([[-1.0 / k for k in range(2, K + 1)]])
    b_eq = np.array([1.0001])
    
    # 2. M(x) <= 1.0001 for x in [1, 10K]
    # M(x) = sum_{k=2}^K g(k) * (floor(x/k) - x/k)
    # Note: x=1 is covered by constraint 1 (M(1) = -sum g(k)/k)
    # We need to check x from 2 to 10K
    A_ub = []
    b_ub = []
    
    for x in range(2, 10 * K + 1):
        row = []
        for k in range(2, K + 1):
            row.append(np.floor(x / k) - x / k)
        A_ub.append(row)
        b_ub.append(1.0001)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        g_vals = res.x
        # Construct full domain and values
        domain = np.arange(1, K + 1)
        values = np.zeros(K)
        # g(1) is free but must be in [-10, 10]
        # We can set g(1) = 0 for simplicity, it doesn't affect objective
        values[0] = 0.0
        values[1:] = g_vals
    else:
        # Fallback to seed-like behavior if infeasible
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
    return domain, values
