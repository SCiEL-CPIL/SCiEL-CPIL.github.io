import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    # Set a deadline guard
    start_time = time.time()
    deadline = 500  # Leave 100s margin
    
    # Choose K. 
    # Based on knowledge, K=50 gave 1.009. 
    # Let's try K=200 or 500.
    # The LP size is (10*K) x K.
    # For K=500, size is 5000x500 = 2.5M elements. Fast.
    # For K=1000, size is 10000x1000 = 10M elements. Still fast.
    # Let's try K=1000.
    
    K = 1000
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to seed if something goes wrong immediately
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct the constraint matrix B
    # B[x-1, j-1] = floor(x/j) - x/j for x in 1..10K, j in 1..K
    # We want to minimize -c^T f subject to B f <= 1
    
    # Vectorized construction
    # x ranges from 1 to 10K
    # j ranges from 1 to K
    x_vals = np.arange(1, 10 * K + 1)[:, np.newaxis]  # shape (10K, 1)
    j_vals = np.arange(1, K + 1)[np.newaxis, :]      # shape (1, K)
    
    # Calculate floor(x/j) - x/j
    # Be careful with floating point precision? 
    # floor(x/j) is integer. x/j is float.
    # The result is in [-1, 0].
    B = np.floor(x_vals / j_vals) - (x_vals / j_vals)
    
    # The constraints are B @ f <= 1
    # We also need bounds -10 <= f <= 10
    bounds = [(-10.0, 10.0)] * K
    
    # Objective: Maximize sum_{k=2}^K (-log(k)/k) * f_k
    # So minimize sum_{k=2}^K (log(k)/k) * f_k
    c = np.zeros(K)
    k_vals = np.arange(1, K + 1)
    # k=1 term is 0
    c[1:] = np.log(k_vals[1:]) / k_vals[1:]
    
    # Solve LP
    # Use HiGHS if available, otherwise CBC
    res = opt.linprog(c, A_ub=B, b_ub=np.ones(10 * K), bounds=bounds, method='highs')
    
    # Check time again
    if time.time() - start_time > deadline:
        # If it's taking too long, we might need to abort, but linprog is synchronous.
        # If it returns, we are good.
        pass

    if res.success:
        f_vals = res.x
    else:
        # If LP fails, fallback to a heuristic or seed
        # Try to use the seed or a simple Mobius approximation
        # For now, return seed to ensure validity
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f_vals.astype(np.float64)
    
    return domain, values
