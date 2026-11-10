# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K: int) -> np.ndarray:
    """
    Compute the Möbius function mu(n) for n = 1..K using a sieve.
    Returns an array of shape (K+1,) where index 0 is unused (set to 0).
    mu(1) = 1, mu(n) = (-1)^r if n is squarefree with r distinct prime factors, else 0.
    """
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin before 630s limit

    # We'll use the full Mobius function up to some K.
    # The key insight: M(x) = sum_{k<=K} mu(k) * floor(x/k)
    # For x <= K, this equals 1 (by the classic identity).
    # For K < x <= 10K, we need to check that M(x) <= 1.
    
    # Let's try K = 200000 (the maximum allowed).
    # But computing M(x) for all x in [1, 10K] = [1, 2000000] with m = K terms
    # would cost K * 10K = 4 * 10^10 operations, which is too slow.
    
    # Wait, let me re-read: "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K] (cost about m * 10K)"
    # So the evaluator does this computation. We just need to return f such that it's feasible.
    # But we need to VERIFY feasibility before returning.
    
    # For the full Mobius function up to K:
    # For x <= K: M(x) = sum_{k<=min(K,x)} mu(k) floor(x/k) = 1 (exactly, by identity)
    # For x > K: M(x) = sum_{k<=K} mu(k) floor(x/k)
    # We need M(x) <= 1 for all x in [1, 10K].
    
    # Actually, the identity sum_{k=1}^x mu(k) floor(x/k) = 1 holds for ALL x >= 1.
    # So for x <= K: M(x) = 1.
    # For x > K: M(x) = sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k)
    # This could be > 1 or < 1 depending on the tail.
    
    # Hmm, this is tricky. Let me think differently.
    
    # Actually, I think the safest approach is to use a SMALL K where we can verify
    # that M(x) <= 1 for all x in [1, 10K], and still get a good score.
    
    # The score is a = -sum_k f(k) log(k) / k.
    # For the full Mobius function: a = -sum_{k=1}^K mu(k) log(k) / k.
    # As K -> infinity, this approaches 1 (the optimum).
    # For small K, it's less than 1 but still reasonable.
    
    # Let me try to find the largest K where we can verify feasibility within time budget.
    # Verification cost: for each x in [1, 10K], compute sum_{k=1}^K mu(k) floor(x/k).
    # That's 10K * K operations = 10 * K^2.
    # For K = 1000: 10 * 10^6 = 10^7 operations (fast)
    # For K = 5000: 10 * 25 * 10^6 = 2.5 * 10^8 (might be slow in pure Python)
    # For K = 10000: 10 * 10^8 = 10^9 (too slow in pure Python)
    
    # But we can vectorize with numpy!
    # For a given x, floor(x/k) for k=1..K can be computed quickly.
    # Actually, we can compute all M(x) at once using a clever approach.
    
    # Alternative: just use a moderate K and trust that the Mobius function works.
    # The known identity guarantees M(x) = 1 for x <= K.
    # For x > K, we need M(x) <= 1.
    
    # Let me try K = 200000 and NOT verify all x, but instead use a mathematical argument
    # or verify a subset.
    
    # Actually, re-reading the problem: the evaluator will check ALL x in [1, 10K].
    # If any M(x) > 1.0001, it's infeasible.
    # So I MUST ensure feasibility.
    
    # Let me think about what happens with the full Mobius function.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k)
    # For x > K, floor(x/k) = x/k - frac(x/k)
    # M(x) = sum_{k=1}^K mu(k) (x/k - frac(x/k))
    #       = x * sum_{k=1}^K mu(k)/k - sum_{k=1}^K mu(k) frac(x/k)
    
    # We know sum_{k=1}^infinity mu(k)/k = 0.
    # So sum_{k=1}^K mu(k)/k ≈ -sum_{k=K+1}^infinity mu(k)/k, which is small.
    
    # This is getting complicated. Let me just use a moderate K and verify.
    
    # Try K = 5000
    K = 5000
    mu = _compute_mobius(K)
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values[0] = 0.0  # f(1) will be adjusted by evaluator
    
    # Verify feasibility: check M(x) <= 1.0001 for all x in [1, 10K]
    # Use vectorized computation
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:  # safety margin
        # Compute M(x) for all x in [1, 10K]
        # M(x) = sum_{k=1}^K mu(k) floor(x/k)
        # Vectorize: for each k, floor(x/k) is a step function
        # Create an array of all x values
        x_vals = np.arange(1, 10 * K + 1, dtype=np.int64)
        
        # Compute M(x) using the fact that floor(x/k) changes at multiples of k
        # For efficiency, compute in chunks
        M = np.zeros(len(x_vals), dtype=np.float64)
        
        for k in range(1, K + 1):
            if mu[k] == 0:
                continue
            # floor(x/k) for x in [1, 10K]
            # This is a step function that increases by 1 at x = k, 2k, 3k, ...
            # We can compute it as: floor(x/k) = x // k
            M += mu[k] * (x_vals // k)
            
            if (k % 500 == 0) and (time.time() - start_time > deadline - 10):
                break
        
        max_M = np.max(M)
        if max_M > 1.0001:
            # Infeasible! Try smaller K
            # Binary search for the largest feasible K
            K_lo, K_hi = 1, K - 1
            best_K = K_lo
            best_mu = None
            
            for K_try in [1000, 2000, 3000, 4000]:
                if K_try > K_hi:
                    continue
                mu_try = _compute_mobius(K_try)
                domain_try = np.arange(1, K_try + 1, dtype=np.int64)
                values_try = mu_try[1:].astype(np.float64)
                values_try[0] = 0.0
                
                # Quick feasibility check
                x_vals_try = np.arange(1, 10 * K_try + 1, dtype=np.int64)
                M_try = np.zeros(len(x_vals_try), dtype=np.float64)
                for k in range(1, K_try + 1):
                    if mu_try[k] == 0:
                        continue
                    M_try += mu_try[k] * (x_vals_try // k)
                
                if np.max(M_try) <= 1.0001:
                    best_K = K_try
                    best_mu = mu_try
                    if time.time() - start_time > deadline - 10:
                        break
            
            if best_mu is not None:
                K = best_K
                mu = best_mu
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = mu[1:].astype(np.float64)
                values[0] = 0.0
            else:
                # Fallback to the original seed values
                domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                return domain, values
        else:
            # Feasible!
            pass
    else:
        # Out of time, fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
