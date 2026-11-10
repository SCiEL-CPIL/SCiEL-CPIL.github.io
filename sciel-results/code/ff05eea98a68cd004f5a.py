# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def compute_mobius(K):
        """Compute Mobius function up to K using a sieve."""
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        is_prime[0] = False
        is_prime[1] = False
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
        return mu

    def check_feasibility_and_score(domain, values, K):
        """
        Simulate the evaluator's steps:
        1. Clip values to [-10, 10].
        2. Adjust f(1) so sum f(k)/k = 0.
        3. Check M(x) <= 1.0001 for x in [1, 10K].
        4. Calculate objective a.
        Returns (is_feasible, a).
        """
        # 1. Clip
        vals = np.clip(values, -10, 10).astype(np.float64)
        
        # 2. Adjust f(1)
        # Note: domain might not contain 1? The problem says "f(1) = 0 before this if 1 is not in the domain".
        # But our domain is 1..K, so 1 is always there.
        # S = sum_k vals[k-1] / k
        k_arr = np.arange(1, K + 1, dtype=np.float64)
        S = np.sum(vals / k_arr)
        vals[0] -= S  # f(1) is at index 0
        
        # 3. Check M(x)
        # M(x) = sum_{k=1}^K vals[k-1] * floor(x/k)
        # We can compute this efficiently.
        # M(x) for x=1..10K.
        # M(x) = sum_{k=1}^K vals[k-1] * floor(x/k)
        
        # Optimization:
        # floor(x/k) is constant on intervals.
        # But simply doing a loop over k and adding vals[k-1] * (x // k) is O(K * 10K).
        # If K=500, 500 * 5000 = 2.5M ops. Fast.
        # If K=1000, 1000 * 10000 = 10M ops. Fast.
        
        max_x = 10 * K
        x_vals = np.arange(1, max_x + 1)
        M = np.zeros(max_x, dtype=np.float64)
        
        for k in range(1, K + 1):
            M += vals[k-1] * (x_vals // k)
            
        if np.any(M > 1.0001):
            return False, 0.0
            
        # 4. Calculate objective
        # a = -sum_k f(k) log(k) / k
        # log(1)=0, so first term is 0.
        a = -np.sum(vals * np.log(k_arr) / k_arr)
        
        return True, a

    def solve_lp(K, mu):
        """
        Solve the LP for g(k) for k=2..K.
        Variables: g(2), ..., g(K).
        Objective: Minimize sum g(k) log(k)/k  (to maximize -sum ...)
        Constraints: sum_{k=2}^K g(k) (floor(x/k) - x/k) <= 1 for x in 1..10K.
        Bounds: g(k) in [-10, 10].
        
        Returns (domain, values, a) if feasible and better, else None.
        """
        n_vars = K - 1
        if n_vars == 0:
            return None
            
        c = np.zeros(n_vars)
        for i, k in enumerate(range(2, K + 1)):
            c[i] = np.log(k) / k
            
        # Build A_ub
        # A_ub[i, j] = floor(x_i / k_j) - x_i / k_j
        # x_i in 1..10K
        # k_j in 2..K
        
        # To save memory and time, we can construct this block by block or just do it if K is small enough.
        # For K=100, A_ub is 1000 x 99. ~100k entries. Trivial.
        # For K=500, A_ub is 5000 x 499. ~2.5M entries. ~20MB. OK.
        # For K=1000, A_ub is 10000 x 999. ~10M entries. ~80MB. OK.
        
        A_ub = np.zeros((10 * K, n_vars))
        b_ub = np.ones(10 * K)
        
        x_vals = np.arange(1, 10 * K + 1)
        k_vars = np.arange(2, K + 1)
        
        # Vectorized construction of A_ub
        # Shape (10K, K-1)
        # floor(x/k) - x/k
        # x_vals[:, None] / k_vars[None, :]
        # This creates a 10K x (K-1) array.
        # For K=1000, 10000 x 999 = 10M floats = 80MB. Acceptable.
        
        ratios = x_vals[:, None] / k_vars[None, :]
        A_ub = np.floor(ratios) - ratios
        
        bounds = [(-10, 10)] * n_vars
        
        # Use HiGHS
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            return None
            
        g = res.x
        # Construct full values array
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0 # Placeholder for f(1), will be calculated in check
        values[1:] = g
        
        # We need to simulate the evaluator to get the final f(1) and check feasibility.
        # The LP solved for g such that M_new(x) <= 1 assuming f(1) is adjusted.
        # But we must ensure the actual clipped/adjusted values satisfy it.
        # The LP constraints were derived from M_new(x) = sum g(k) h_k(x) <= 1.
        # This is exactly the M(x) after adjustment.
        # So if LP succeeds, it should be feasible.
        # However, we should double check with the simulation to be safe against numerical errors.
        
        is_feas, a = check_feasibility_and_score(np.arange(1, K + 1), values, K)
        
        if is_feas:
            return np.arange(1, K + 1), values, a
        else:
            return None

    best_result = None
    best_a = -np.inf

    # Strategy:
    # 1. Try pure Mobius for small K (fast, no LP).
    # 2. Try LP for small K to see if we can beat Mobius.
    # 3. Try larger K with Mobius if LP is too slow or not improving.
    
    # Step 1: Pure Mobius
    for K in [50, 100, 200, 500, 1000, 2000, 5000]:
        if time.time() > deadline - 30:
            break
            
        mu = compute_mobius(K)
        values = mu[1:K+1].astype(np.float64)
        domain = np.arange(1, K + 1)
        
        is_feas, a = check_feasibility_and_score(domain, values, K)
        
        if is_feas and a > best_a:
            best_a = a
            best_result = (domain, values)
            # print(f"Mobius K={K}, a={a}")
            
        if time.time() > deadline - 60:
            break

    # Step 2: LP for small K
    # The LP might yield a better 'a' for small K by optimizing the tail.
    # Let's try K=10, 20, 50.
    for K in [10, 20, 50, 100]:
        if time.time() > deadline - 30:
            break
            
        mu = compute_mobius(K) # Just to have it, though LP doesn't strictly need it if we define bounds
        res = solve_lp(K, mu)
        
        if res is not None:
            domain, values, a = res
            if a > best_a:
                best_a = a
                best_result = (domain, values)
                # print(f"LP K={K}, a={a}")

    if best_result is not None:
        return best_result
    else:
        # Fallback to baseline
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values


# EVOLVE-BLOCK-END
