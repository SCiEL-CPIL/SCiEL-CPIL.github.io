<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Use a moderate K to ensure feasibility and speed
    K = 1000
    
    # Compute Möbius function mu(k) for k = 1..K
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    # Check feasibility
    # Compute S = sum f(k)/k
    S = np.sum(values / domain)
    
    # Effective f
    f_eff = values.copy()
    f_eff[0] = values[0] - S # f(1) is at index 0
    
    # Compute M(x) for x in [1, 10K]
    max_x = 10 * K
    M_max = -np.inf
    # Vectorized check for all x is O(K * 10K)
    # We can do it in chunks or just accept the cost for K=1000 (10^7 ops)
    # 10^7 numpy ops is fast.
    
    # M(x) = sum f_eff[k] * floor(x/k)
    # We can compute this for all x at once?
    # x is 1..max_x
    # k is 1..K
    # floor(x/k) is a matrix of shape (max_x, K)
    # This might be too much memory (10000 * 1000 = 10^7 floats = 80MB). OK.
    
    x_vals = np.arange(1, max_x + 1, dtype=np.int64)
    k_vals = domain.astype(np.int64)
    
    # Compute floor division matrix
    # Broadcasting: x_vals[:, None] / k_vals[None, :]
    floors = np.floor(x_vals[:, None] / k_vals[None, :]).astype(np.int64)
    
    # M(x) = floors @ f_eff
    M_vals = floors @ f_eff
    
    M_max = np.max(M_vals)
    
    if M_max > 1.0001:
        # Scale down to be feasible
        # If we scale f(k) for k>1 by c, and adjust f(1)?
        # Simplest: scale all f(k) by c?
        # If we scale all f by c, S scales by c, f'(1) scales by c.
        # M(x) scales by c.
        # a scales by c.
        # So we can just scale the whole function.
        c = 1.0 / M_max
        values = values * c
        # Note: This reduces the score, but ensures feasibility.
        
    return domain, values

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 550  # safety margin
    
    # Use a larger K for better approximation
    K = 200000
    
    # Compute Möbius function mu(k) for k = 1..K using sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    # Check time - if too slow, reduce K
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Fallback to smaller K
        K = 10000
        mu = np.ones(K + 1, dtype=np.int64)
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, K + 1):
            if is_prime[i]:
                for j in range(i, K + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
                i2 = i * i
                if i2 <= K:
                    for j in range(i2, K + 1, i2):
                        mu[j] = 0
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K + 1].astype(np.float64)
    
    # Compute S = sum f(k)/k
    S = np.sum(values / domain)
    
    # Effective f: f(1) is adjusted
    f_eff = values.copy()
    f_eff[0] = values[0] - S
    
    # Compute M(x) for x in [1, 10K] efficiently
    max_x = 10 * K
    M_vals = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each k, add f_eff[k-1] * floor(x/k) to M(x) for x in [k, max_x]
    # floor(x/k) = q for x in [k*q, k*(q+1)-1]
    # Use difference array for range updates
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        if time.time() > deadline - 60:
            break
        fk = f_eff[k - 1]
        if fk == 0:
            continue
        # For q = 1, 2, ..., floor(max_x/k)
        max_q = max_x // k
        for q in range(1, max_q + 1):
            lo = k * q
            hi = min(k * (q + 1) - 1, max_x)
            diff[lo] += fk * q
            diff[hi + 1] -= fk * q
    
    # Convert difference array to M_vals
    np.cumsum(diff, out=M_vals)
    M_vals = M_vals[1:max_x + 1]  # M(1) to M(max_x)
    
    M_max = np.max(M_vals)
    
    if M_max > 1.0001:
        c = 1.0 / M_max
        values = values * c
    
    return domain, values

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the nested loop over k and q is O(K * 10K/k) = O(10K * H(K)) which for K=200000 is about 200000 * 10 * ln(200000) ≈ 200000 * 10 * 12 = 24 million iterations in Python. That's too slow.

I need a vectorized approach. Let me think differently.

M(x) = sum_{k=1}^{K} f(k) * floor(x/k)

For a fixed x, floor(x/k) changes at k = floor(x/j) for j = 1, 2, ... So we can compute M(x) in O(sqrt(x)) time. But we need all x from 1 to 10K.

Alternative: precompute for each possible value of floor(x/k). Note that floor(x/k) = q iff x in [k*q, k*(q+1)-1].

Let me use a different approach: for each q from 1 to 10 (since x <= 10K and k >= 1, floor(x/k) <= 10K), the values of x where floor(x/k) = q for some k... this is getting complex.

Actually, let me just use the matrix approach but with smaller K. The score for K=1000 was 0.077. Let me try K=5000 with the matrix approach.

For K=5000, max_x = 50000. Matrix is 50000 x 5000 = 250 million floats = 2GB. Too much.

Let me chunk the x values. Process x in chunks of 5000. For each chunk, compute a 5000 x K matrix. For K=20000, that's 5000 x 20000 = 100 million = 800MB per chunk. Still too much.

For K=10000, chunk size 1000: 1000 x 10000 = 10 million = 80MB per chunk. 100 chunks total. Each chunk does 10^7 multiply-adds. Total 10^9 ops. In numpy, this should take a few seconds.

Let me try K=10000 with chunked computation.

