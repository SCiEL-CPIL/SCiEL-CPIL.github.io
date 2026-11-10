import numpy as np
from scipy.optimize import linprog
import math

def prime_number_theorem_partial_function():
    K = 100
    # Variables are f(2), ..., f(K)
    n_vars = K - 1
    
    # Objective: maximize - sum f(k) log(k)/k
    # linprog minimizes c^T x. So c = sum log(k)/k
    c = np.zeros(n_vars)
    for i, k in enumerate(range(2, K + 1)):
        c[i] = math.log(k) / k
        
    # Constraints: sum_{k=2}^K f(k) ( floor(x/k) - x/k ) <= 1
    # A_ub x <= b_ub
    A_ub = np.zeros((10 * K, n_vars))
    b_ub = np.ones(10 * K)
    
    for x in range(1, 10 * K + 1):
        for i, k in enumerate(range(2, K + 1)):
            A_ub[x - 1, i] = math.floor(x / k) - x / k
            
    # Bounds: f(k) in [-10, 10]
    bounds = [(-10, 10) for _ in range(n_vars)]
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to mu(k)
        # Compute mu
        mu = np.zeros(K + 1, dtype=int)
        mu[1] = 1
        for i in range(2, K + 1):
            if mu[i] == 0: # prime
                for j in range(i, K + 1, i):
                    mu[j] = -mu[j]
                    if j % (i * i) == 0:
                        mu[j] = 0
        # But mu(1) is handled by evaluator.
        # We return f(2)..f(K) from LP or mu.
        # If LP fails, use mu.
        domain = np.array(list(range(2, K + 1)), dtype=np.int64)
        values = mu[2:K + 1].astype(np.float64)
        # We must include 1 in domain? 
        # "domain holds distinct positive integers"
        # "f(1) is replaced by ... if 1 is not in domain, f(1)=0"
        # It is better to include 1 in domain so we can set it?
        # No, f(1) is always adjusted.
        # If 1 is not in domain, f(1)=0 initially.
        # Then f(1)_new = 0 - S = -S.
        # If 1 is in domain, f(1)_old = v. f(1)_new = v - S.
        # The value of f(1) returned doesn't matter for the score, only for the constraint check?
        # "M(x) is computed ... if any M(x) > 1.0001"
        # M(x) uses the *replaced* f(1).
        # So the value we return for 1 does not affect M(x) directly, except through S?
        # S = sum f(k)/k.
        # If 1 is in domain, S includes f(1)/1.
        # If 1 is not in domain, S does not include f(1) (which is 0).
        # But f(1)_new is calculated.
        # Let's just include 1 in domain and set f(1)=0.
        # Then S = sum_{k=2} f(k)/k.
        # f(1)_new = -S.
        # This is consistent.
        
        domain = np.array(list(range(1, K + 1)), dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = mu[2:K + 1].astype(np.float64)
        return domain, values

    # Extract solution
    f_vals = res.x
    domain = np.array(list(range(1, K + 1)), dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    values[0] = 0.0 # f(1) placeholder, will be adjusted by evaluator
    values[1:] = f_vals
    
    return domain, values
