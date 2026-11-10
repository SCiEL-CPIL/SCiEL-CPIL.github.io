# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Use a smoothed kernel f(k) = -c/k * exp(-k/K_smooth) for k > 1,
    which approximates a Dirichlet convolution structure that keeps M(x) bounded.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

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
        k_arr = np.arange(1, K + 1, dtype=np.float64)
        S = np.sum(vals / k_arr)
        vals[0] -= S  # f(1) is at index 0
        
        # 3. Check M(x)
        max_x = 10 * K
        x_vals = np.arange(1, max_x + 1)
        M = np.zeros(max_x, dtype=np.float64)
        
        for k in range(1, K + 1):
            M += vals[k-1] * (x_vals // k)
            
        if np.any(M > 1.0001):
            return False, 0.0
            
        # 4. Calculate objective
        a = -np.sum(vals * np.log(k_arr) / k_arr)
        
        return True, a

    def try_smoothed_kernel(K, K_smooth, c_scale):
        """
        Try a smoothed kernel: f(k) = -c_scale * exp(-k/K_smooth) for k > 1.
        The 1/k factor is implicit in the objective weighting.
        """
        domain = np.arange(1, K + 1)
        values = np.zeros(K, dtype=np.float64)
        
        # For k > 1: f(k) = -c_scale * exp(-k / K_smooth)
        ks = np.arange(2, K + 1, dtype=np.float64)
        values[1:] = -c_scale * np.exp(-ks / K_smooth)
        
        # f(1) will be set by the evaluator to satisfy sum f(k)/k = 0
        # But we need to check if the resulting M(x) is feasible
        
        is_feas, a = check_feasibility_and_score(domain, values, K)
        
        return (domain, values, a) if is_feas else None

    def try_mobius(K):
        """Try pure Möbius function."""
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
        
        domain = np.arange(1, K + 1)
        values = mu[1:K+1].astype(np.float64)
        
        is_feas, a = check_feasibility_and_score(domain, values, K)
        
        return (domain, values, a) if is_feas else None

    best_result = None
    best_a = -np.inf

    # Strategy 1: Try smoothed kernels with various parameters
    # f(k) = -c * exp(-k/K_smooth) for k > 1
    # The key insight: this creates a smooth decay that should keep M(x) bounded
    
    # First, let's understand the constraint better.
    # M(x) = sum_{k=1}^K f(k) * floor(x/k)
    # After adjustment: M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k) + [sum_{k=2}^K f(k)] * x/K * (1 - 1/K)
    # Wait, let me recalculate.
    
    # Actually: M(x) = f(1)*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k)
    # = f(1)*x + sum_{k=2}^K f(k)*floor(x/k)
    # = f(1)*x + sum_{k=2}^K f(k)*(x/k + (floor(x/k) - x/k))
    # = x*(f(1) + sum_{k=2}^K f(k)/k) + sum_{k=2}^K f(k)*(floor(x/k) - x/k)
    # = 0 + sum_{k=2}^K f(k)*(floor(x/k) - x/k)   [since f(1) = -sum_{k=2}^K f(k)/k]
    
    # So M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
    # Note that (floor(x/k) - x/k) is in [-1, 0) for all x, k.
    # So if f(k) >= 0 for all k > 1, then M(x) <= 0 < 1. Feasible!
    # But then a = -sum f(k)*log(k)/k <= 0. Not good.
    
    # If f(k) < 0 for some k, then f(k)*(floor(x/k) - x/k) > 0 (since both negative).
    # So we need f(k) to be negative to get positive M(x) contributions, but we need M(x) <= 1.
    
    # Let's think differently. The optimal solution from LP had specific structure.
    # Let me try: f(k) = -c/k for k > 1 (harmonic-like decay)
    
    # Actually, let's just try various parameterizations and see what works.
    
    # Try 1: Pure Möbius (baseline)
    for K in [100, 200, 500, 1000, 2000, 5000, 10000, 200000]:
        if time.time() > deadline - 30:
            break
        
        result = try_mobius(K)
        if result is not None:
            domain, values, a = result
            if a > best_a:
                best_a = a
                best_result = (domain, values)

    # Try 2: Smoothed exponential kernel with negative values
    # f(k) = -c * exp(-k/K_smooth) for k > 1
    # We need to find c such that max_x M(x) <= 1
    
    for K in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 60:
            break
        
        for K_smooth in [K/10, K/5, K/2, K, 2*K, 5*K, 10*K]:
            if time.time() > deadline - 60:
                break
            
            # Binary search for c such that max M(x) = 1
            # Start with c = 0.1 and increase
            
            lo, hi = 0.0, 10.0
            best_c = 0.0
            best_a_for_K = -np.inf
            best_vals_for_K = None
            best_dom_for_K = None
            
            for _ in range(20):  # Binary search iterations
                mid = (lo + hi) / 2
                
                domain = np.arange(1, K + 1)
                values = np.zeros(K, dtype=np.float64)
                ks = np.arange(2, K + 1, dtype=np.float64)
                values[1:] = -mid * np.exp(-ks / K_smooth)
                
                is_feas, a = check_feasibility_and_score(domain, values, K)
                
                if is_feas:
                    if a > best_a_for_K:
                        best_a_for_K = a
                        best_c = mid
                        best_vals_for_K = values.copy()
                        best_dom_for_K = domain.copy()
                    lo = mid
                else:
                    hi = mid
            
            if best_a_for_K > best_a and best_vals_for_K is not None:
                best_a = best_a_for_K
                best_result = (best_dom_for_K, best_vals_for_K)

    # Try 3: Power law decay f(k) = -c * k^(-alpha)
    for K in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:
            break
        
        for alpha in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            if time.time() > deadline - 30:
                break
            
            lo, hi = 0.0, 10.0
            best_c = 0.0
            best_a_for_K = -np.inf
            best_vals_for_K = None
            best_dom_for_K = None
            
            for _ in range(20):
                mid = (lo + hi) / 2
                
                domain = np.arange(1, K + 1)
                values = np.zeros(K, dtype=np.float64)
                ks = np.arange(2, K + 1, dtype=np.float64)
                values[1:] = -mid * np.power(ks, -alpha)
                
                is_feas, a = check_feasibility_and_score(domain, values, K)
                
                if is_feas:
                    if a > best_a_for_K:
                        best_a_for_K = a
                        best_c = mid
                        best_vals_for_K = values.copy()
                        best_dom_for_K = domain.copy()
                    lo = mid
                else:
                    hi = mid
            
            if best_a_for_K > best_a and best_vals_for_K is not None:
                best_a = best_a_for_K
                best_result = (best_dom_for_K, best_vals_for_K)

    # Try 4: Mixed approach - use Möbius-like sign pattern with smooth magnitudes
    for K in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:
            break
        
        # Compute Möbius
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
        
        # Use sign of mu, smooth magnitude
        for K_smooth in [K/5, K, 5*K, 20*K]:
            if time.time() > deadline - 30:
                break
            
            lo, hi = 0.0, 10.0
            best_c = 0.0
            best_a_for_K = -np.inf
            best_vals_for_K = None
            best_dom_for_K = None
            
            for _ in range(15):
                mid = (lo + hi) / 2
                
                domain = np.arange(1, K + 1)
                values = np.zeros(K, dtype=np.float64)
                ks = np.arange(2, K + 1, dtype=np.float64)
                # Use sign of mu, smooth magnitude
                mu_sign = np.sign(mu[2:K+1].astype(np.float64))
                values[1:] = mid * mu_sign * np.exp(-ks / K_smooth)
                
                is_feas, a = check_feasibility_and_score(domain, values, K)
                
                if is_feas:
                    if a > best_a_for_K:
                        best_a_for_K = a
                        best_c = mid
                        best_vals_for_K = values.copy()
                        best_dom_for_K = domain.copy()
                    lo = mid
                else:
                    hi = mid
            
            if best_a_for_K > best_a and best_vals_for_K is not None:
                best_a = best_a_for_K
                best_result = (best_dom_for_K, best_vals_for_K)

    if best_result is not None:
        return best_result
    else:
        # Fallback: Möbius for K=200
        mu = np.ones(201, dtype=np.int8)
        is_prime = np.ones(201, dtype=bool)
        primes = []
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, 201):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > 200:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        
        domain = np.arange(1, 201)
        values = mu[1:201].astype(np.float64)
        return domain, values


# EVOLVE-BLOCK-END
