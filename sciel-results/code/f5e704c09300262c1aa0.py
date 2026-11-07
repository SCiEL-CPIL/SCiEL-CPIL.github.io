import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    temp = n
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
        result *= (temp + 1)  # sigma(p) = p + 1
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

def _sigma_sieve_multiplicative(N):
    """Compute sigma(n) for all n <= N using multiplicative sieve over primes."""
    sig = [0] * (N + 1)
    sig[1] = 1
    # smallest prime factor sieve
    spf = [0] * (N + 1)
    for i in range(2, N + 1):
        if spf[i] == 0:
            for j in range(i, N + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    for i in range(2, N + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            # i = m * p, and p divides m, so p^a || m for some a >= 1
            # sigma(i) = sigma(m) * (p^(a+2) - 1) / (p^(a+1) - 1)
            # We need to find the power of p in m
            # Actually, let's use: if p^a || m, then i = p^(a+1) * (m/p^a)
            # sigma(i) = sigma(m/p^a) * (p^(a+2)-1)/(p-1)
            # and sigma(m) = sigma(m/p^a) * (p^(a+1)-1)/(p-1)
            # So sigma(i) = sigma(m) * (p^(a+2)-1) / (p^(a+1)-1)
            # Find a: highest power of p dividing m
            temp_m = m
            a = 0
            while temp_m % p == 0:
                temp_m //= p
                a += 1
            # a >= 1 since p divides m
            # sigma(i) = sigma(m) * (p^(a+2)-1) / (p^(a+1)-1)
            pa1 = p ** (a + 1)
            pa2 = pa1 * p
            sig[i] = sig[m] * (pa2 - 1) // (pa1 - 1)
        else:
            # p does not divide m, so i = m * p with gcd(m,p)=1
            # sigma(i) = sigma(m) * (p+1)
            sig[i] = sig[m] * (p + 1)
    return sig

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 125.0  # Safety margin from 150s
    
    # Use multiplicative sieve - much faster than O(N log N) divisor sum
    # Try progressively larger N until time runs out
    N = 10 * 10**6
    sig = _sigma_sieve_multiplicative(N)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        # Too slow, try smaller
        N = 5 * 10**6
        sig = _sigma_sieve_multiplicative(N)
        elapsed = time.time() - start_time
    
    if time.time() - start_time > deadline - 30:
        return {'M': 1, 'k': [1]}
    
    # Now group k by M = k * sigma(k)
    groups = defaultdict(list)
    max_M = 10**18
    
    for k in range(1, N + 1):
        m = k * sig[k]
        if m <= max_M:
            groups[m].append(k)
        else:
            if k * k > max_M:
                break
    
    # Find best group
    best_M = 0
    best_k_list = []
    
    for m, k_list in groups.items():
        if len(k_list) > len(best_k_list):
            best_M = m
            best_k_list = k_list
    
    # Now try to find additional k values for best_M by checking all divisors
    if best_M > 0:
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            # Get all divisors of best_M
            divs = _divisors(best_M)
            # For each divisor k, check if k <= sqrt(best_M) and k * sigma(k) == best_M
            sqrt_M = math.isqrt(best_M)
            found_k = set(best_k_list)
            for k in divs:
                if k <= sqrt_M and k not in found_k:
                    # Check if k * sigma(k) == best_M
                    s = _sigma(k)
                    if k * s == best_M:
                        found_k.add(k)
            
            if len(found_k) > len(best_k_list):
                best_k_list = list(found_k)
    
    # Additional search: look for more solutions by checking candidates
    # near the best_M in a targeted way
    if best_M > 0 and time.time() - start_time < deadline - 10:
        found_k = set(best_k_list)
        sqrt_M = math.isqrt(best_M)
        # Check divisors that might have been missed
        # Try to find k where k * sigma(k) = best_M by factoring approach
        # For each prime factor of best_M, try combinations
        # This is expensive, so only do a limited search
        # Check k values that are divisors of best_M
        divs = _divisors(best_M)
        for k in divs:
            if k <= sqrt_M and k not in found_k:
                s = _sigma(k)
                if k * s == best_M:
                    found_k.add(k)
        
        if len(found_k) > len(best_k_list):
            best_k_list = list(found_k)
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
