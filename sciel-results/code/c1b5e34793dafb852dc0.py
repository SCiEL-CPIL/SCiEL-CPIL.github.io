import math
import time
from collections import defaultdict

def _sigma_trial(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n <= 1:
        return n
    result = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
            result *= (d ** (exp + 1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)
    return result

def _divisors(n):
    """Generate all divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin from 150s limit
    
    # Phase 1: Sieve sigma up to a reasonable limit
    # We want to find M with many k such that k*sigma(k)=M
    # Larger N gives more chances for collisions, but slower sieve
    N = 5 * 10**6
    
    sigma = [0] * (N + 1)
    sig = sigma
    n_limit = N
    
    # Direct sieve for sigma: O(N log N)
    # This is the bottleneck. For N=5e6, it's ~5e6 * H(5e6) ~ 5e6 * 15 = 7.5e7 ops
    # In Python, this might take 5-10 seconds.
    for i in range(1, n_limit + 1):
        for j in range(i, n_limit + 1, i):
            sig[j] += i
            
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Fallback: return a known small solution
        return {'M': 333312, 'k': [336, 372, 434]}

    # Phase 2: Group k by M = k * sigma(k)
    groups = defaultdict(list)
    max_M = 10**18
    
    for k in range(1, N + 1):
        m = k * sig[k]
        if m <= max_M:
            groups[m].append(k)
        else:
            # Since sigma(k) >= k, m >= k^2. If k^2 > max_M, we can stop.
            # However, sigma(k) can be much larger than k, so m grows faster.
            # But if k is large, m is likely > max_M.
            # Actually, the condition is just m <= max_M.
            # We can break if k * k > max_M? No, sigma(k) >= k is true, so m >= k^2.
            # If k^2 > max_M, then m > max_M for all subsequent k?
            # Not necessarily, but sigma(k) is generally >= k.
            # Let's just check the condition.
            pass

    # Find best group
    best_M = 0
    best_k_list = []
    
    for m, k_list in groups.items():
        if len(k_list) > len(best_k_list):
            best_M = m
            best_k_list = k_list
            
    # Phase 3: Expand search for best_M
    # We might have missed some k > N.
    # Or we can check divisors of best_M to see if there are more solutions.
    if best_M > 0:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Get all divisors of best_M
            # Note: best_M can be up to 10^18, so divisors can be many but usually manageable
            divs = _divisors(best_M)
            sqrt_M = math.isqrt(best_M)
            found_k = set(best_k_list)
            
            # Sort divisors to check smaller ones first (more likely to be valid k <= sqrt(M))
            # Actually, we just need to check all divisors k <= sqrt(M)
            for k in divs:
                if k <= sqrt_M and k not in found_k:
                    # Check if k * sigma(k) == best_M
                    s = _sigma_trial(k)
                    if k * s == best_M:
                        found_k.add(k)
            
            if len(found_k) > len(best_k_list):
                best_k_list = list(found_k)
                
    # Ensure types are correct
    # M must be int, k must be list of ints
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
