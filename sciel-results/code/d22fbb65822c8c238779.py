# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Strategy: Use Möbius function for all square-free k up to K, but with a different
    # optimization approach. The key insight is that the constraint M(x) <= 1.0001 is
    # tight, and we want to maximize a = -sum f(k)*log(k)/k.
    
    # The Möbius function approach gives M(x) = sum_{k<=x} mu(k) * floor(x/k) which
    # relates to the count of square-free numbers. But we need to verify the constraint.
    
    # Let's try a different approach: use only primes with value -1, which gives
    # M(x) = -sum_{p<=x} floor(x/p) + f1*x where f1 is chosen to make sum f(k)/k = 0.
    # This might allow larger K while keeping M(x) bounded.
    
    # Actually, let's stick with the Möbius approach but optimize the computation.
    # The current approach computes M(x) for all x in [1, 10K] which is O(m * 10K).
    # For K=100000, m~60000, 10K=1000000, so ~6*10^10 operations - too slow.
    
    # Better approach: Use the fact that M(x) = sum_k f(k) * floor(x/k)
    # For the Möbius function, M(x) = sum_{k<=x} mu(k) * floor(x/k)
    # This can be computed more efficiently using the identity:
    # sum_{k=1}^n mu(k) * floor(n/k) = 1 for n >= 1 (this is a known identity)
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually, sum_{d|n} mu(d) = 1 if n=1, else 0.
    # And sum_{k=1}^n mu(k) * floor(n/k) = 1 for all n >= 1.
    # This is because sum_{k=1}^n mu(k) * floor(n/k) = sum_{k=1}^n mu(k) * sum_{j=1}^{floor(n/k)} 1
    # = sum_{j=1}^n sum_{k=1}^{floor(n/j)} mu(k) ... no that's not right either.
    
    # Let me think again. sum_{k=1}^n mu(k) * floor(n/k)
    # = sum_{k=1}^n mu(k) * sum_{j: k*j <= n} 1
    # = sum_{m=1}^n sum_{k|m} mu(k)
    # = sum_{m=1}^n [m==1]  (since sum_{d|m} mu(d) = 1 if m=1, else 0)
    # = 1
    
    # So if we use f(k) = mu(k) for k >= 2, and f(1) is determined by the constraint
    # sum_k f(k)/k = 0, then:
    # M(x) = f(1)*x + sum_{k=2}^x mu(k) * floor(x/k)
    # We know sum_{k=1}^x mu(k) * floor(x/k) = 1, so sum_{k=2}^x mu(k) * floor(x/k) = 1 - mu(1)*floor(x/1) = 1 - x
    # Therefore M(x) = f(1)*x + (1 - x) = f(1)*x + 1 - x = (f(1) - 1)*x + 1
    # For this to be <= 1.0001 for all x in [1, 10K], we need f(1) - 1 <= 0, i.e., f(1) <= 1.
    # Also M(x) should be >= something (no lower bound given, but let's check).
    # Actually the constraint is only M(x) > 1.0001 causes infeasibility.
    # So we need (f(1) - 1)*x + 1 <= 1.0001 for all x in [1, 10K].
    # If f(1) < 1, then (f(1)-1) is negative, so M(x) decreases with x, max at x=1: M(1) = f(1) - 1 + 1 = f(1) <= 1.0001.
    # If f(1) = 1, M(x) = 1 for all x.
    # If f(1) > 1, M(x) increases with x, max at x=10K: (f(1)-1)*10K + 1 <= 1.0001, so f(1) <= 1 + 0.0001/(10K).
    
    # Now, f(1) is determined by sum_k f(k)/k = 0.
    # sum_k f(k)/k = f(1)/1 + sum_{k=2}^K mu(k)/k = 0
    # So f(1) = -sum_{k=2}^K mu(k)/k
    
    # We need to check if f(1) <= 1 (or more precisely, if the constraint is satisfied).
    # sum_{k=2}^K mu(k)/k is related to 1/sum_{k=1}^K mu(k)/k - 1 = 1/(sum_{k=1}^K mu(k)/k) ... no.
    # Actually, sum_{k=1}^K mu(k)/k approximates 1/pi^2 * something? No, sum mu(k)/k^2 = 1/zeta(2) = 6/pi^2.
    # sum mu(k)/k does not converge, it oscillates.
    
    # Let's just compute it numerically for various K and check if the constraint holds.
    
    # For the objective: a = -sum_k f(k) * log(k) / k
    # = -f(1)*log(1)/1 - sum_{k=2}^K mu(k) * log(k) / k
    # = -sum_{k=2}^K mu(k) * log(k) / k  (since log(1)=0)
    
    # We want to maximize a, i.e., minimize sum_{k=2}^K mu(k) * log(k) / k.
    # This is related to the derivative of log(sum mu(k)/k^s) at s=1... not exactly.
    
    # Let's try different K values and see which gives the best feasible solution.
    
    best_result = None
    best_a = -np.inf
    
    for K in [50000, 30000, 20000, 10000, 5000, 2000, 1000, 500]:
        if time.time() > deadline:
            break
        
        # Compute Möbius function up to K using linear sieve
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
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
        
        # Domain: k where mu[k] != 0, k >= 2
        mask = (mu != 0)
        mask[0] = False
        mask[1] = False
        domain = np.where(mask)[0]
        values = mu[domain].astype(np.float64)
        m = len(domain)
        
        if m < 1:
            continue
        
        # Compute S = sum f(k)/k over domain
        S = np.sum(values / domain)
        
        # f(1) = -S
        f1 = -S
        
        # Check constraint: M(x) = (f1 - 1)*x + 1 for x in [1, 10K]
        # Max of M(x) is at x=10K if f1 > 1, or at x=1 if f1 <= 1
        max_x = 10 * K
        if f1 > 1:
            max_M = (f1 - 1) * max_x + 1
        else:
            max_M = f1  # at x=1
        
        if max_M > 1.0001:
            continue
        
        # Feasible! Compute objective
        a = -np.sum(values * np.log(domain.astype(np.float64)) / domain)
        
        if a > best_a:
            best_a = a
            best_result = (domain, values)
        
        if time.time() > deadline:
            break
    
    if best_result is not None:
        return best_result
    
    return fallback()


# EVOLVE-BLOCK-END
