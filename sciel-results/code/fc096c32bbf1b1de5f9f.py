import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Phase 1: Sieve sigma up to N using NumPy for speed
    # 50 million is a good balance of memory (~400MB) and coverage
    N = 50000000
    
    # Compute sigma via sieve
    # sigma[k] = sum of divisors of k
    sigma = np.zeros(N + 1, dtype=np.int64)
    # We iterate i from 1 to N, adding i to sigma[j] for all j multiple of i
    # This is O(N log N) but vectorized operations are fast
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 60:
        # Emergency fallback if sieve takes too long (unlikely with NumPy)
        # Use a smaller N or return a known good solution
        return _fallback_result()

    # Compute M_k = k * sigma(k)
    # k ranges from 1 to N
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:N+1]
    
    # Filter M_vals <= 10^18
    max_M = 10**18
    mask = M_vals <= max_M
    valid_M_vals = M_vals[mask]
    valid_k_vals = k_vals[mask]
    
    # Find the M with the most occurrences
    # Using np.unique to count
    # This might be memory heavy if there are many unique M's, but usually manageable
    # Alternative: use a dict if counts are sparse, but np.unique is faster for dense data
    # However, we need to retrieve the k's for the best M.
    
    # Let's find the top candidates first
    # We can use a histogram or just sort
    # Sorting 50M integers is fast (~1-2s)
    sorted_M = np.sort(valid_M_vals)
    
    # Find counts
    # We can use np.bincount if the range was small, but M is up to 10^18.
    # So we use np.unique with return_counts
    unique_M, counts = np.unique(valid_M_vals, return_counts=True)
    
    # Find the index of the max count
    max_count = np.max(counts)
    best_M_idx = np.argmax(counts)
    best_M = unique_M[best_M_idx]
    
    # Get the k's for this best_M
    # We need to find all k in valid_k_vals where M_vals == best_M
    # This is a linear scan or we can use the inverse mapping from np.unique
    # np.unique doesn't return the indices directly in a useful way for this without extra work.
    # But we can just do:
    k_candidates = valid_k_vals[valid_M_vals == best_M]
    
    # It's possible that a different M has the same count, or more if we consider extensions.
    # Let's collect top 10 M's to check for extensions
    top_indices = np.argsort(-counts)[:10]
    top_Ms = unique_M[top_indices]
    
    best_ks = list(k_candidates)
    best_count = len(best_ks)
    
    # Phase 2: Extension search for top candidates
    # Check if there are k > N such that k * sigma(k) = M
    # We use the factorization method for these specific M's
    
    for M_val in top_Ms:
        if time.time() > deadline - 10:
            break
            
        # Get existing k's for this M
        # We already have them for the best_M, but for others we need to fetch
        if M_val == best_M:
            existing_ks = set(best_ks)
            ext_ks = list(best_ks)
        else:
            ks_for_M = valid_k_vals[valid_M_vals == M_val]
            existing_ks = set(ks_for_M)
            ext_ks = list(ks_for_M)
        
        # Find divisors k > N and k <= sqrt(M)
        sqrt_M = math.isqrt(M_val)
        
        # Factorize M to get divisors
        divisors = _get_divisors(M_val)
        
        for k in divisors:
            if k > N and k <= sqrt_M and k not in existing_ks:
                expected_sigma = M_val // k
                # Verify sigma(k)
                if _sigma_by_factorization(k) == expected_sigma:
                    ext_ks.append(k)
                    existing_ks.add(k)
                    
        if len(ext_ks) > best_count:
            best_M = M_val
            best_ks = sorted(ext_ks)
            best_count = len(ext_ks)
            
    return _finalize(best_M, best_ks)

def _fallback_result():
    # A known good result from the previous program
    M = 333312
    ks = [336, 372, 434]
    return {'M': M, 'k': ks}

def _finalize(M, ks):
    """Ensure correct types and constraints."""
    M = int(M)
    ks = sorted(set(int(k) for k in ks))
    # Verify k <= isqrt(M)
    sqrt_M = math.isqrt(M)
    ks = [k for k in ks if k <= sqrt_M]
    return {'M': M, 'k': ks}

def _get_divisors(n):
    """Get all divisors of n using factorization."""
    if n <= 0:
        return []
    
    factors = _factorize(n)
    
    # Generate all divisors from prime factorization
    divisors = [1]
    for p, e in factors:
        current = list(divisors)
        p_power = 1
        for i in range(e):
            p_power *= p
            for d in current:
                divisors.append(d * p_power)
    
    return divisors

def _factorize(n):
    """Factorize n using trial division with wheel. Returns list of (prime, exponent)."""
    if n <= 1:
        return []
    
    factors = []
    temp = n
    
    # Divide by 2
    if temp % 2 == 0:
        count = 0
        while temp % 2 == 0:
            temp //= 2
            count += 1
        factors.append((2, count))
    
    # Divide by 3
    if temp % 3 == 0:
        count = 0
        while temp % 3 == 0:
            temp //= 3
            count += 1
        factors.append((3, count))
    
    # Wheel: 2, 3, 4, 6, 8, 12 pattern (skip multiples of 2 and 3)
    d = 5
    wheel = [2, 4, 2, 4, 2, 4, 6, 2, 6]  # increments to skip multiples of 2,3
    wheel_idx = 0
    
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors.append((d, count))
        
        d += wheel[wheel_idx]
        wheel_idx = (wheel_idx + 1) % len(wheel)
    
    if temp > 1:
        factors.append((temp, 1))
    
    return factors

def _sigma_by_factorization(n):
    """Compute sigma(n) by factoring n."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    factors = _factorize(n)
    result = 1
    for p, e in factors:
        # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
        p_power = p ** (e + 1)
        result *= (p_power - 1) // (p - 1)
    
    return result
