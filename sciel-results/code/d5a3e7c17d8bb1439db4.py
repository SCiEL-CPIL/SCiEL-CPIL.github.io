# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the objective a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1 for all x in [1, 10K].

    Strategy:
      1. Compute the Möbius function mu(k) for k up to K.
      2. Start with f(k) = mu(k) for all k in [1, K].
      3. Clip values to [-10, 10] (mu is already in {-1,0,1}).
      4. Compute S = sum_k f(k)/k, then set f(1) -= S to make sum_k f(k)/k = 0.
      5. Check feasibility: M(x) <= 1.0001 for all x in [1, 10K].
         M(x) = sum_k f(k) * floor(x/k).
         Note: after the f(1) adjustment, M(x) = (f(1) + sum_{k>=2} f(k)*floor(x/k))
              = -S + sum_{k>=2} mu(k)*floor(x/k)
         Actually let's recompute properly.
      6. If infeasible, we can try to scale down or use a smaller K.
      7. Return (domain, values).

    The key insight: for the full Möbius function, sum_{k=1}^{floor(x/k')} mu(k') * floor(x/k') 
    is NOT 1. The identity is sum_{d|n} mu(d) = 1 if n=1 else 0.
    But M(x) = sum_k mu(k) * floor(x/k) = sum_{n<=x} sum_{d|n, d in domain} mu(d).
    For full domain [1,K], M(x) = sum_{n<=x} (1 if n=1 and 1<=K else 0) = 1 for x>=1.
    Wait, that's only if K >= x. For x > K, the inner sum is over d|n with d<=K.
    
    Actually, sum_{k=1}^{K} mu(k) * floor(x/k) = sum_{n=1}^{x} sum_{k|n, k<=K} mu(k).
    For n <= K: inner sum = 1 if n=1 else 0. So contributes 1.
    For n > K: inner sum = sum_{k|n, k<=K} mu(k), which is not necessarily 0.
    
    So M(x) = 1 + sum_{n=K+1}^{x} sum_{k|n, k<=K} mu(k) for x > K.
    
    This can be large. Let's just compute it numerically.
    
    For the objective: a = -sum_k mu(k)*log(k)/k (after the f(1) adjustment).
    The sum -sum_{k=1}^K mu(k)*log(k)/k converges to 1 as K->infinity.
    For finite K, it's less than 1. The f(1) adjustment subtracts S from f(1),
    which adds S*log(1)/1 = 0 to the objective. So the objective is unchanged by the f(1) fix!
    Wait: a = -sum_k f(k)*log(k)/k. log(1) = 0, so f(1) doesn't contribute.
    So a = -sum_{k=2}^K f(k)*log(k)/k = -sum_{k=2}^K mu(k)*log(k)/k.
    
    We want to maximize this. Larger K gives a value closer to 1 (from below).
    
    The constraint M(x) <= 1.0001 for all x in [1, 10K] is the binding one.
    For x <= K, M(x) = 1 (exactly, since the full divisor sum up to K covers all divisors for n<=K).
    Wait, that's not right either. Let me recompute.
    
    M(x) = sum_{k=1}^K mu(k) * floor(x/k).
    floor(x/k) = number of multiples of k in [1, x] = number of n in [1,x] with k|n.
    So M(x) = sum_{k=1}^K mu(k) * |{n in [1,x] : k|n}| = sum_{n=1}^x |{k in [1,K] : k|n}|.
    For n <= K: |{k in [1,K] : k|n}| includes all divisors of n, so sum = sum_{d|n} mu(d) = 1 if n=1 else 0.
    For n > K: |{k in [1,K] : k|n}| is the number of divisors of n that are <= K.
    So M(x) = 1 + sum_{n=K+1}^x (number of divisors of n that are <= K, weighted by mu).
    Actually: M(x) = sum_{n=1}^x sum_{k|n, k<=K} mu(k).
    For n<=K: = sum_{k|n} mu(k) = [n==1].
    For n>K: = sum_{k|n, k<=K} mu(k), which is generally not 0.
    
    So M(x) for x > K can deviate from 1. We need M(x) <= 1.0001.
    
    Let's just compute M(x) for all x in [1, 10K] and check.
    The cost is O(K * 10K / log K) roughly, or more precisely sum_{k=1}^K floor(10K/k) = 10K * H_K ~ 10K log K.
    For K=200000, that's about 200000 * 10 * 12 = 24 million operations. Should be fine.
    
    Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Try increasing K values, starting from a reasonable one
    # For each K, compute mu, build f, check feasibility, compute objective
    # Return the best feasible result
    
    best_result = None
    best_score = -np.inf
    
    # Start with K values that are likely to work
    # The constraint M(x) <= 1.0001 is tight. Let's try K=10000 first, then increase.
    K_values = [5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_values:
        if time.time() > deadline - 30:
            break
        
        # Compute Möbius function up to K
        mu = np.ones(K + 1, dtype=np.int64)
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        
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
        
        # Build f: f(k) = mu(k) for k in [1, K]
        # domain = [1, 2, ..., K], values = mu[1], mu[2], ..., mu[K]
        # But we only need non-zero mu values for efficiency
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:].astype(np.float64)
        
        # Clip to [-10, 10] (already satisfied since mu in {-1,0,1})
        values = np.clip(values, -10.0, 10.0)
        
        # Compute S = sum_k f(k)/k
        S = np.sum(values / domain)
        
        # Adjust f(1): f(1) -= S
        # domain[0] = 1, so values[0] is f(1)
        values[0] -= S
        
        # Re-clip after adjustment
        values = np.clip(values, -10.0, 10.0)
        
        # Check feasibility: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
        # M(x) = sum_{k=1}^K f(k) * floor(x/k)
        # We can compute this as: for each k, f(k) contributes f(k) to M(x) for x in [k, 2k-1], 
        # f(k)*2 for x in [2k, 3k-1], etc.
        # More efficiently: M(x) = sum_{k=1}^K f(k) * floor(x/k)
        # Let's compute M(x) for all x in [1, 10K] using a difference array approach.
        
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        # For each k, f(k) * floor(x/k) is a step function.
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1].
        # So f(k) * floor(x/k) adds f(k)*j to M(x) for x in [j*k, min((j+1)*k-1, max_x)].
        # We can use a difference array: for each k, for each j >= 1,
        #   start = j*k, end = min((j+1)*k - 1, max_x)
        #   if start <= max_x:
        #       M_diff[start] += f(k)*j
        #       M_diff[end+1] -= f(k)*j  (if end+1 <= max_x)
        
        M_diff = np.zeros(max_x + 2, dtype=np.float64)
        
        for k in range(1, K + 1):
            fk = values[k - 1]
            if fk == 0.0:
                continue
            j = 1
            while j * k <= max_x:
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                M_diff[start] += fk * j
                if end + 1 <= max_x:
                    M_diff[end + 1] -= fk * j
                j += 1
        
        M = np.cumsum(M_diff)
        M = M[1:]  # M[x] for x in [1, max_x]
        
        # Check feasibility
        if np.max(M) > 1.0001:
            # Infeasible, skip this K
            continue
        
        # Compute objective: a = -sum_k f(k) * log(k) / k
        # f(1) doesn't contribute since log(1) = 0
        log_domain = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_domain / domain)
        
        if a > best_score:
            best_score = a
            best_result = (domain.copy(), values.copy())
        
        # If we've reached a score very close to 1, we can stop
        if best_score > 0.999:
            break
    
    # If no feasible result was found, fall back to the seed
    if best_result is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_result


# EVOLVE-BLOCK-END
