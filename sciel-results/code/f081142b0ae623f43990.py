import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to
    sum_k f(k)/k = 0 and M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    
    Strategy:
    Use the Mobius function mu(k) as the base signal, which naturally satisfies the
    Mertens identity sum_{k<=x} mu(k) floor(x/k) = 1 (for infinite support).
    For finite K, we compute the deviation D(x) and scale the function to ensure
    M(x) <= 1.0001 while maximizing the score.
    """
    start_time = time.time()
    deadline = 600.0
    
    # We want to use a large K to approximate the infinite series sum mu(k)log(k)/k -> -1.
    # However, we must ensure the computation fits in the time budget.
    # The dominant cost is computing M(x) for x in [1, 10K].
    # Naive O(10K * K) is too slow for large K.
    # We can use the sieve-like approach: G = zeros(10K+1); for k in 1..K: G[k::k] += mu[k].
    # This is O(10K * log(10K)) roughly, which is fast.
    
    # Let's try K = 50000. 10K = 500000.
    # Sieve for mu: O(K log log K).
    # G computation: sum_{k=1}^K (10K/k) ~ 10K log K ~ 500000 * 10 = 5e6 ops. Very fast.
    
    K = 50000
    
    # 1. Compute Mobius function mu[1..K]
    mu = np.ones(K + 1, dtype=np.int8)
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
    mu[1] = 1
    
    if time.time() - start_time > deadline - 10:
        # Fallback to small K if sieve was unexpectedly slow
        K = 1000
        mu = np.ones(K + 1, dtype=np.int8)
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
        mu[1] = 1

    # 2. Compute S(x) = sum_{k=1}^K mu(k) floor(x/k) for x in 1..10K
    N_max = 10 * K
    G = np.zeros(N_max + 1, dtype=np.float64)
    
    # Efficient update: for each k, add mu[k] to indices k, 2k, 3k...
    # We only care about k where mu[k] != 0
    for k in range(1, K + 1):
        if mu[k] != 0:
            G[k::k] += float(mu[k])
    
    # S(x) is the cumulative sum of G[1..x]
    # Note: G[i] stores the contribution to M(i) from the "step" at i?
    # No. M(x) = sum_{k=1}^K f(k) floor(x/k).
    # floor(x/k) is the number of multiples of k <= x.
    # So M(x) = sum_{j=1}^x (sum_{k|j} f(k)).
    # Let H(j) = sum_{k|j} f(k). Then M(x) = sum_{j=1}^x H(j).
    # If f(k) = c * mu(k), then H(j) = c * sum_{k|j} mu(k) = c * (1 if j=1 else 0).
    # Wait, sum_{d|n} mu(d) = 1 if n=1, 0 if n>1.
    # So H(1) = c, H(j) = 0 for j > 1.
    # Then M(x) = c for all x >= 1.
    # This is the infinite case.
    
    # But we have finite K.
    # f(k) = c * mu(k) for k <= K.
    # H(j) = c * sum_{d|j, d<=K} mu(d).
    # For j <= K, sum_{d|j} mu(d) is 1 if j=1, 0 otherwise. So H(j) is c if j=1, 0 else.
    # For j > K, the sum is partial.
    
    # Let's re-evaluate M(x) using the G array approach.
    # G[i] = sum_{k|i, k<=K} mu(k).
    # Then M(x) = sum_{j=1}^x G[j].
    
    S_vals = np.cumsum(G[1:]) # S_vals[i-1] = M(i) for i=1..N_max
    
    # 3. Determine scaling factor c
    # The evaluator:
    # 1. Clips f to [-10, 10].
    # 2. Computes S_sum = sum_{k=1}^K f(k)/k.
    # 3. f_new(1) = f(1) - S_sum.
    # 4. M_new(x) = f_new(1)*x + sum_{k=2}^K f(k) floor(x/k).
    #    Note: sum_{k=1}^K f(k) floor(x/k) = M_old(x).
    #    M_new(x) = M_old(x) - S_sum * floor(x/1) = M_old(x) - S_sum * x.
    #    Wait, f(1) is replaced. The term for k=1 in M_old is f(1)*x.
    #    In M_new, the term for k=1 is f_new(1)*x.
    #    So M_new(x) = M_old(x) - (f(1) - f_new(1))*x = M_old(x) - S_sum * x.
    
    # We set f(k) = c * mu(k).
    # M_old(x) = c * S_mu(x) where S_mu(x) = sum_{k=1}^K mu(k) floor(x/k).
    # S_sum = c * sum_{k=1}^K mu(k)/k. Let R = sum_{k=1}^K mu(k)/k.
    # M_new(x) = c * S_mu(x) - c * R * x = c * (S_mu(x) - R * x).
    
    # We need M_new(x) <= 1.0001 for all x in [1, 10K].
    # Let D(x) = S_mu(x) - R * x.
    # We need c * max(D(x)) <= 1.0001 (assuming c > 0).
    # Also we want to maximize a = -sum f(k) log(k)/k = -c * sum mu(k) log(k)/k.
    # Let L = sum mu(k) log(k)/k.
    # a = -c * L.
    # We know sum mu(k) log(k)/k -> -1 as K->inf. So L is negative.
    # So a = c * |L|. We want c as large as possible.
    
    # Compute R
    indices = np.arange(1, K + 1, dtype=np.float64)
    R = np.sum(mu[1:] / indices)
    
    # Compute D(x)
    x_vals = np.arange(1, N_max + 1, dtype=np.float64)
    D_vals = S_vals - R * x_vals
    
    max_D = np.max(D_vals)
    
    # If max_D <= 0, we can set c very large?
    # But f(k) must be in [-10, 10].
    # mu(k) is in {-1, 0, 1}. So |f(k)| = c.
    # So c <= 10.
    
    c_max_clip = 10.0
    if max_D > 1e-12:
        c_max_constraint = 1.0001 / max_D
    else:
        c_max_constraint = np.inf
    
    c = min(c_max_clip, c_max_constraint)
    
    # Construct f
    f = c * mu[1:].astype(np.float64)
    
    # Check if c is effectively 0 or negative (should not happen)
    if c <= 0:
        # Fallback to small safe values
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([0.0, -0.1], dtype=np.float64)
        return domain, values
    
    # Construct domain and values
    # We include all k from 1 to K where mu(k) != 0?
    # The problem says "domain holds distinct positive integers".
    # It doesn't say we must include all.
    # Including zeros doesn't help and increases m.
    # But m * 10K is the cost. If we include all K, m=K.
    # If we only include non-zeros, m ~ 6K/pi^2.
    # Let's include only non-zeros to be safe on memory/time if m matters.
    # But the evaluator computes M(x) for x in 1..10K.
    # If we omit k where f(k)=0, it doesn't change M(x).
    
    mask = mu[1:] != 0
    domain = np.arange(1, K + 1, dtype=np.int64)[mask]
    values = f[mask]
    
    return domain, values

if __name__ == "__main__":
    d, v = prime_number_theorem_partial_function()
    print(d[:10], v[:10])
