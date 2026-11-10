import numpy as np
import math
import time
import sys

def _sigma_sieve(N):
    """Compute sigma(k) for k=1..N using a sieve."""
    sigma = np.ones(N + 1, dtype=np.int64)
    for j in range(1, N + 1):
        # Add j to sigma[j], sigma[2j], ...
        sigma[j::j] += j
    return sigma

def _factorize(n):
    """Factorize n into a dict of prime: exponent."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2 # Simple optimization, skip even numbers after 2
        if d == 2: d = 3
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _get_divisors(factors):
    """Generate all divisors from factorization."""
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        p_pow = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _sigma_from_factors(factors):
    """Compute sigma(n) from its factorization."""
    s = 1
    for p, e in factors.items():
        s *= (p ** (e + 1) - 1) // (p - 1)
    return s

def _verify_M(M):
    """Find all k such that k*sigma(k)=M and k <= sqrt(M)."""
    if M <= 0:
        return []
    sqrt_M = int(math.isqrt(M))
    factors = _factorize(M)
    divisors = _get_divisors(factors)
    valid_k = []
    for k in divisors:
        if k > sqrt_M:
            continue
        # Check if k is a divisor (it is by construction)
        # Check if sigma(k) == M // k
        # We need sigma(k). We can compute it by factorizing k or using the property.
        # Since k divides M, we can get factors of k from factors of M? 
        # Not directly, but we can factorize k quickly since k <= sqrt(M).
        # Or better: we already have factors of M. 
        # Actually, computing sigma(k) via factorization of k is fast enough for k <= 10^9.
        f_k = _factorize(k)
        s_k = _sigma_from_factors(f_k)
        if k * s_k == M:
            valid_k.append(k)
    return valid_k

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, 20s buffer
    
    # Seed solution
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Search parameters
    # We want to find M with many solutions.
    # Let's sieve up to N.
    # N=10^6 is safe. Let's try 2*10^6 if time permits, but 10^6 is robust.
    N = 10**6
    
    try:
        sigma = _sigma_sieve(N)
        # Compute M values
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma
        
        # Filter M <= 10^18
        max_M = 10**18
        mask = M_vals <= max_M
        M_vals = M_vals[mask]
        k_vals = k_vals[mask]
        
        if len(M_vals) == 0:
            return {'M': best_M, 'k': best_k}
        
        # Find most frequent M
        unique_M, counts = np.unique(M_vals, return_counts=True)
        
        # Get indices of top candidates
        # We want the top 20 most frequent M
        top_indices = np.argsort(counts)[::-1][:20]
        
        candidates = []
        for idx in top_indices:
            M = int(unique_M[idx])
            count = int(counts[idx])
            # Only consider if count > current best count
            if count > len(best_k):
                candidates.append((M, count))
        
        # Evaluate candidates
        for M, count in candidates:
            if time.time() > deadline:
                break
            
            valid_k = _verify_M(M)
            if len(valid_k) > len(best_k):
                best_M = M
                best_k = valid_k
                
    except Exception as e:
        # Fallback to seed if error
        pass
        
    # Ensure types are Python int and list
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}

if __name__ == "__main__":
    res = construct()
    print(res)
