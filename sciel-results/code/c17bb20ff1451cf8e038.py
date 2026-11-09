# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # We want to approximate the Mobius function behavior.
    # We select K to balance accuracy and computation time.
    # Based on knowledge, K=1000 is good. Let's try K=2000 if fast enough.
    # We will start with K=1000 and potentially increase if time permits,
    # but a single well-chosen K is often better than a complex loop.
    # Let's fix K=2000 for a stronger result.
    K = 2000
    
    # 1. Compute Mobius function mu(k) for k = 1..K
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(K)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            # For mu calculation: if p^2 divides n, mu(n)=0.
            # We can compute mu via sieve.
            # mu(1) = 1
            # mu(p) = -1
            # mu(p*q) = 1
            # mu(p^2) = 0
            
    # Efficient Mu Sieve
    mu = np.ones(K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # 2. Select domain: square-free integers (mu(k) != 0)
    # We must include k=1 if mu(1)!=0 (which it is).
    domain_candidates = np.where(mu != 0)[0]
    domain_candidates = domain_candidates[domain_candidates >= 1]
    
    # Ensure 1 is in the domain for the normalization step to work as described
    if 1 not in domain_candidates:
        domain_candidates = np.insert(domain_candidates, 0, 1)
        
    # Sort domain
    domain_candidates = np.sort(domain_candidates)
    
    # If K is too large or time is tight, we might need to trim.
    # But K=2000 is small.
    
    m = len(domain_candidates)
    
    # 3. Setup LP
    # Variables: f(k) for k in domain_candidates where k > 1
    # f(1) is determined by: f(1) = - sum_{k>1} f(k)/k
    
    # Index 0 in domain_candidates is k=1 (since we ensured it)
    # Variables correspond to domain_candidates[1:]
    
    if m < 2:
        # Fallback if something went wrong
        return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)
        
    k_vars = domain_candidates[1:] # k > 1
    n_vars = len(k_vars)
    
    # Objective: Maximize - sum_{k>1} f(k) * log(k) / k
    # linprog minimizes c^T x. So we minimize - (objective)
    # c_k = - ( - log(k)/k ) = log(k)/k
    c = np.log(k_vars.astype(np.float64)) / k_vars.astype(np.float64)
    
    # Constraints:
    # For each x in 1..10K:
    # sum_{k>1} f(k) * (floor(x/k) - x/k) <= 1.0001
    # Note: f(1) * floor(x/1) + sum_{k>1} f(k) floor(x/k) <= 1.0001
    # -x * sum_{k>1} f(k)/k + sum_{k>1} f(k) floor(x/k) <= 1.0001
    # sum_{k>1} f(k) (floor(x/k) - x/k) <= 1.0001
    
    X_max = 10 * K
    # We don't need to check every x if the function is monotonic or has structure?
    # M(x) is a step function. The maximum of M(x) over an interval [j, j+1) is at x=j.
    # Actually M(x) is constant on [k, k+1) for integer k? No, floor(x/k) changes at multiples.
    # M(x) is a step function that is right-continuous?
    # floor(x/k) is constant for x in [j*k, (j+1)*k - 1] (integers).
    # For real x, floor(x/k) is constant on [j*k, (j+1)*k).
    # So M(x) is constant on intervals between multiples of any k.
    # The critical points are integers. The maximum of M(x) for x in [1, X_max]
    # will occur at some integer x?
    # Let's check. M(x) = sum f(k) floor(x/k).
    # Between integers, floor(x/k) is constant. So M(x) is constant.
    # At integer boundaries, it jumps.
    # Since we require M(x) <= 1.0001 for ALL x, and M(x) is constant on intervals,
    # we only need to check the values at the right-end of the intervals or just integer values?
    # Actually, if M(x) <= C for all integers, does it hold for all reals?
    # M(x) is non-decreasing or non-increasing? No, f(k) can be negative.
    # However, M(x) is a step function. Its value on [n, n+1) is M(n).
    # So checking integers x=1..X_max is sufficient to cover all real x in [1, X_max].
    
    # Sparse matrix construction
    # Rows: x = 1 to X_max
    # Cols: k_vars (indices 0 to n_vars-1)
    
    # To save memory and time, we can observe that for large x, the term x/k is large.
    # The constraint is sum f(k) (floor(x/k) - x/k) <= 1.0001.
    # Let g_k(x) = floor(x/k) - x/k. This is in [-1, 0).
    
    # We will build the sparse matrix A.
    rows = []
    cols = []
    data = []
    
    # Optimization: Only store non-zero entries.
    # For a given x, and k in k_vars:
    # entry = floor(x/k) - x/k
    # This is non-zero unless x/k is integer (then 0).
    # So we skip if x % k == 0.
    
    # This loop might be slow in Python. Let's vectorize or use efficient iteration.
    # X_max = 20000, n_vars ~ 1200. Total pairs ~ 24 million.
    # Python loop over 24M items is too slow (~10-20s).
    # We need a faster way to build A or solve.
    
    # Alternative: Iterate over k_vars and update rows?
    # Or use numpy to generate indices?
    
    # Let's try to reduce X_max.
    # Does M(x) oscillate significantly for large x?
    # The sum approximates 1. The error is bounded.
    # Perhaps we don't need all x up to 10K?
    # The contract says: "M(x) ... is computed for every integer x in [1, 10K]".
    # So we MUST satisfy constraints for all x in that range.
    
    # We can build A column by column.
    # For each k in k_vars:
    #   for x in 1..X_max:
    #      if x % k != 0:
    #         A[x-1, col_idx] = floor(x/k) - x/k
    # This is still O(X_max * n_vars).
    
    # Let's check if we can skip many constraints.
    # If f(k) are small, the sum is small.
    # But we want to maximize the objective, so f(k) will be pushed to bounds or active constraints.
    
    # Let's try to build the matrix using scipy sparse hstack or similar.
    # Actually, we can just use a dense array if it's small enough?
    # 20000 x 1200 = 24M floats = 192 MB. That's acceptable.
    # Dense matrix construction in numpy might be faster than sparse if we fill it carefully.
    
    A_dense = np.zeros((X_max, n_vars), dtype=np.float64)
    
    # Vectorized fill?
    # x is column vector (X_max, 1)
    # k is row vector (1, n_vars)
    # floor(x/k) - x/k
    # This creates a (X_max, n_vars) array. 24M elements.
    # Memory: 24M * 8 bytes = 192 MB.
    # Time: Vectorized numpy ops are fast.
    
    x_col = np.arange(1, X_max + 1, dtype=np.float64).reshape(-1, 1)
    k_row = k_vars.astype(np.float64).reshape(1, -1)
    
    # Calculate A
    # We need to avoid division by zero? k_vars >= 2, so no.
    A_vals = np.floor(x_col / k_row) - x_col / k_row
    
    # Where x % k == 0, A_vals is 0.
    # We can just use A_vals directly.
    
    # Check memory/time.
    # If this is too slow or memory heavy, we fall back to sparse.
    # But 192MB is fine.
    
    b = np.full(X_max, 1.0001, dtype=np.float64)
    
    # Bounds for f(k)
    # The evaluator clips to [-10, 10].
    # We should enforce these in the LP to ensure the solution is valid for the evaluator.
    # If LP solution is within [-10, 10], clipping does nothing.
    # If it exceeds, the evaluator clips, which changes the objective and constraints.
    # It is safer to constrain the LP to [-10, 10].
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve LP
    # Maximize c^T v  =>  Minimize -c^T v
    res = linprog(-c, A_ub=A_vals, b_ub=b, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to seed-like solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    f_vars = res.x
    
    # Construct full values
    f_1 = -np.sum(f_vars / k_vars)
    
    values = np.empty(m, dtype=np.float64)
    values[0] = f_1
    values[1:] = f_vars
    
    # Clip values to [-10, 10] as per evaluator logic, 
    # although LP bounds should have prevented this.
    values = np.clip(values, -10.0, 10.0)
    
    # Re-normalize? 
    # The evaluator does: f(1) = f(1) - S, where S = sum f(k)/k.
    # If we used the LP constraint derived from this normalization, 
    # and then clip, the sum might no longer be 0.
    # The evaluator will re-normalize.
    # So we just return the values. The evaluator handles the rest.
    
    return domain_candidates, values

# EVOLVE-BLOCK-END
