# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # Maximum K to consider
    K_max = 2000
    
    # Compute Mobius function up to K_max
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes to get primes
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, K_max + 1, i):
                is_prime[j] = False
    
    # Compute Mobius function
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^k if n is a product of k distinct primes
    mu = np.ones(K_max + 1, dtype=np.int8)
    for p in primes:
        for j in range(p, K_max + 1, p):
            mu[j] *= -1
        for j in range(p*p, K_max + 1, p*p):
            mu[j] = 0
            
    best_score = -np.inf
    best_domain = None
    best_values = None

    # Try different K values
    for K in [10, 50, 100, 200, 500, 1000, 2000]:
        if time.time() > deadline:
            break
        if K > K_max:
            continue
            
        f = mu[1:K+1].astype(np.float64)
        
        # Precompute sums for efficiency
        # S2 = sum_{k=2}^K f(k)/k
        k_arr = np.arange(1, K+1, dtype=np.float64)
        S2 = np.sum(f[1:] / k_arr[1:])
        
        # Check feasibility
        # M(x) = -x * S2 + sum_{k=2}^K f(k) * floor(x/k)
        # We need M(x) <= 1.0001 for all x in [1, 10K]
        # We can sample x or check all x. 10K is at most 20000.
        # Checking all x is O(10K * K) which is 20000 * 2000 = 40M ops. Might be slow in Python loop.
        # Vectorize:
        # M(x) for all x is a matrix product?
        # Let A be matrix where A[x, k] = floor(x/k) for x in 1..10K, k in 2..K.
        # This is too big to store explicitly (20000 x 2000 = 40M floats = 320MB).
        # We can compute M(x) in chunks or use a smarter method.
        
        # Alternative: Just check a subset of x?
        # The maximum of M(x) likely occurs at points where many floor(x/k) change, i.e., x = k * m.
        # Or we can just compute for all x if K is small.
        
        max_M = -np.inf
        limit_x = 10 * K
        if limit_x <= 5000:
            x_vals = np.arange(1, limit_x + 1)
            # Compute M(x)
            # M(x) = -x * S2 + sum_{k=2}^K f(k) * floor(x/k)
            # We can compute sum part using a loop over k, adding f(k) * floor(x/k)
            M_vals = -x_vals * S2
            for k in range(2, K+1):
                M_vals += f[k-1] * (x_vals // k)
            max_M = np.max(M_vals)
        else:
            # For large K, check a sample of x
            # The worst case is likely when x is large
            x_sample = np.linspace(1, limit_x, 1000).astype(int)
            M_vals = -x_sample * S2
            for k in range(2, K+1):
                M_vals += f[k-1] * (x_sample // k)
            max_M = np.max(M_vals)
            
        if max_M > 1.0001:
            alpha = 1.0001 / max_M
            f_scaled = alpha * f
            # Recompute score with scaled f
            # Score = -sum f(k) log(k) / k
            # f(1) term is 0 in score
            score = -np.sum(f_scaled[1:] * np.log(k_arr[1:]) / k_arr[1:])
        else:
            score = -np.sum(f[1:] * np.log(k_arr[1:]) / k_arr[1:])
            
        if score > best_score:
            best_score = score
            best_domain = np.arange(1, K+1, dtype=np.int64)
            best_values = f.copy() if max_M <= 1.0001 else (alpha * f).copy()

    if best_domain is None:
        # Fallback to seed
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
