import numpy as np

def prime_number_theorem_partial_function():
    K = 200000
    M = 50000
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    values = np.zeros(K, dtype=np.float64)
    # c_1 is 0
    # c_k = -10 for 2 <= k <= M
    values[1:M] = -10.0
    # c_k = 10 for M < k <= K
    values[M:K] = 10.0
    
    return domain, values
