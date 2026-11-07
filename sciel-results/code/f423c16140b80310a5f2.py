"""Find M with many k such that k*sigma(k) = M, using a linear sieve for sigma."""
import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    # We'll enumerate k up to K and compute sigma(k) via a linear sieve.
    # K is chosen to fit within time budget.
    # Start with a reasonable K, increase if time permits.
    
    # Strategy: linear sieve for sigma(n)
    # sigma is multiplicative. For a linear sieve:
    # sigma(p) = 1 + p
    # sigma(p^a) = (p^(a+1) - 1)/(p - 1)
    # sigma(n) = product over prime powers
    
    # Linear sieve approach:
    # spf[i] = smallest prime factor
    # For each i, we can compute sigma using the smallest prime factor decomposition.
    
    # Actually, let's use a standard linear sieve that tracks:
    # - spf: smallest prime factor
    # - For computing sigma, we track the highest power of spf in n and the "remaining" part
    
    # Alternative: use a sieve that computes sigma directly.
    # sigma(n) = sum of divisors. We can use:
    # Initialize sigma[1] = 1
    # For each prime p, for multiples of p: add p^j to sigma
    
    # Most efficient: use the multiplicative property with a linear sieve.
    
    # Let's use this approach:
    # spf[i] = smallest prime factor of i
    # For the linear sieve, we track for each number n:
    #   lp[n] = largest power of spf[n] dividing n (i.e., if n = p^a * m with p not dividing m, lp = p^a)
    #   Then sigma[n] = sigma[lp[n]] * sigma[n // lp[n]]
    #   sigma[lp[n]] = 1 + p + p^2 + ... + p^a where lp[n] = p^a
    
    # Let's implement this.
    
    def run_sieve(K):
        """Compute sigma(k) for k=1..K using linear sieve. Returns list sigma[0..K]."""
        spf = [0] * (K + 1)
        lp = [1] * (K + 1)  # lp[i] = highest power of spf[i] dividing i
        sigma = [0] * (K + 1)
        sigma[1] = 1
        
        primes = []
        for i in range(2, K + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                lp[i] = i
                sigma[i] = 1 + i
            for p in primes:
                if p > spf[i]:
                    break
                if i * p > K:
                    break
                spf[i * p] = p
                if p == spf[i]:
                    # p divides i, so lp[i*p] = lp[i] * p
                    lp[i * p] = lp[i] * p
                    # sigma[i*p] = sigma[i // lp[i]] * sigma[lp[i] * p]
                    # sigma[lp[i] * p] = sigma[lp[i]] + p^a * p where a is exponent in lp[i]
                    # Actually: if lp[i] = p^a, then lp[i*p] = p^(a+1)
                    # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                    # sigma[i*p] = sigma[i // lp[i]] * (sigma[lp[i]] + lp[i] * p)
                    # But sigma[lp[i]] = 1 + p + ... + p^a, and p^(a+1) = lp[i] * p
                    # So sigma[lp[i]*p] = sigma[lp[i]] + lp[i]*p
                    sigma[i * p] = sigma[i // lp[i]] * (sigma[lp[i]] + lp[i] * p)
                    break
                else:
                    # p does not divide i, so spf[i*p] = p, lp[i*p] = p
                    lp[i * p] = p
                    sigma[i * p] = sigma[i] * (1 + p)
        
        return sigma
    
    # Try increasing K until we run out of time
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Start with K=2_000_000 and increase
    K = 2_000_000
    while K <= 10_000_000 and time.time() < deadline - 10:
        sigma = run_sieve(K)
        
        # Now compute f(k) = k * sigma(k) for k=1..K and find collisions
        # Use a dict: M -> list of k's
        from collections import defaultdict
        freq = defaultdict(list)
        
        for k in range(1, K + 1):
            M_val = k * sigma[k]
            freq[M_val].append(k)
        
        # Find M with most preimages
        max_count = 0
        max_M = None
        max_ks = None
        for M_val, ks in freq.items():
            if len(ks) > max_count:
                max_count = len(ks)
                max_M = M_val
                max_ks = ks
        
        if max_count > best_count:
            best_count = max_count
            best_M = max_M
            best_k = max_ks
        
        # Increase K
        K = min(K * 2, 10_000_000)
        if K >= 10_000_000:
            break
        
        if time.time() < deadline - 15:
            continue
        else:
            break
    
    # Verify and format result
    if best_count <= 3:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure all k_i satisfy k_i <= isqrt(best_M) and k_i * sigma(k_i) = best_M
    # The sieve computed sigma for k up to K, and we used those values, so this should hold.
    # But let's double check: k_i must be <= isqrt(M)
    isq = isqrt(best_M)
    valid_ks = [k for k in best_k if k <= isq]
    
    if len(valid_ks) < 2:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in valid_ks]}

# EVOLVE-BLOCK-END
