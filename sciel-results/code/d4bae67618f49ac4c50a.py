import math
import time
from collections import defaultdict

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 150s kill
    
    # Known solution: M = 333312 with k = [336, 372, 434]
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy 1: Find M as product of many p*(p+1) for consecutive primes
    # g(p) = p*(p+1) for prime p
    # We want M = prod of many such terms, each term = p_i*(p_i+1)
    # M must be <= 10^18
    
    # Generate primes up to some limit
    # For p*(p+1) <= 10^18, p <= ~10^9
    # But we need products of many such terms
    
    # Let's try: M = p1*(p1+1) * p2*(p2+1) * ... * pn*(pn+1)
    # Each factor is ~p^2, so product of n factors ~ (p_avg)^{2n}
    # For M <= 10^18, if p_avg ~ 1000, then 2n*log10(1000) = 6n <= 18, so n <= 3
    # If p_avg ~ 100, then 2n*2 = 4n <= 18, so n <= 4
    # If p_avg ~ 10, then 2n*1 = 2n <= 18, so n <= 9
    
    # Let's try small primes: 2,3,5,7,11,13,17,19,23,...
    # g(2)=6, g(3)=12, g(5)=30, g(7)=56, g(11)=132, g(13)=182, g(17)=306, g(19)=380, g(23)=552
    # Product of first n:
    # n=1: 6
    # n=2: 72
    # n=3: 2160
    # n=4: 120960
    # n=5: 15966720
    # n=6: 2905943040
    # n=7: 889218569040
    # n=8: 337903056235200
    # n=9: 186522487041830400
    # n=10: *g(29)=29*30=870 -> 162274563726392448000 > 10^18
    
    # So with first 9 primes, M = 186522487041830400, and k = [2,3,5,7,11,13,17,19,23]
    # Check: each k <= sqrt(M)? sqrt(186522487041830400) ~ 431,882,600
    # All primes up to 23 are << 431 million, so yes!
    
    # Let's verify this and also try other combinations
    
    # First, verify the 9-prime solution
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    M_candidate = 1
    for p in small_primes:
        M_candidate *= p * (p + 1)
    
    if M_candidate <= 10**18:
        # Verify all k satisfy k*sigma(k) = M_candidate
        # For prime p, sigma(p) = 1 + p = p+1
        # So k*sigma(k) = p*(p+1) which divides M_candidate
        # But we need k*sigma(k) = M_candidate exactly, not just divides!
        
        # Wait, the condition is k_i * sigma(k_i) = M, not that M is a product.
        # So for each k in the list, we need k*sigma(k) = M exactly.
        # That means all k must have the SAME value of k*sigma(k) = M.
        
        # So M = p1*(p1+1) = p2*(p2+1) = ... which is impossible for distinct primes!
        
        # I was wrong. The condition is that each k_i satisfies k_i * sigma(k_i) = M.
        # So we need multiple k with the same k*sigma(k) value.
        
        # Let me reconsider. For the known solution M=333312:
        # k=336: sigma(336) = sigma(2^4 * 3 * 7) = (1+2+4+8+16)*(1+3)*(1+7) = 31*4*8 = 992
        # 336*992 = 333312 ✓
        # k=372: sigma(372) = sigma(2^2 * 3 * 31) = (1+2+4)*(1+3)*(1+31) = 7*4*32 = 896
        # 372*896 = 333312 ✓
        # k=434: sigma(434) = sigma(2 * 7 * 31) = (1+2)*(1+7)*(1+31) = 3*8*32 = 768
        # 434*768 = 333312 ✓
        
        # So I need to find M where multiple k have k*sigma(k) = M.
        
        pass
    
    # Strategy: Use sieve to compute sigma(k) for k up to N, then group by k*sigma(k)
    # The current approach with N=2,000,000 found 3 solutions.
    # Let's increase N to find more.
    
    # Memory: sigma array of size N+1. For N=5,000,000, that's ~40MB for int array.
    # Time: O(N log N) for the sieve.
    
    # Let's try N = 5,000,000
    N = 5000000
    
    # Use array module for memory efficiency
    from array import array
    sigma_arr = array('Q', [0]) * (N + 1)  # Unsigned long long
    
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma_arr[m] += d
    
    groups = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        M_val = k * sigma_arr[k]
        if M_val <= 10**18:
            groups[M_val].append(k)
            if len(groups[M_val]) > best_count:
                best_count = len(groups[M_val])
                best_M = M_val
                best_k = groups[M_val][:]
    
    # Also try a targeted search: for specific M values, find all k
    # that satisfy k*sigma(k) = M.
    
    # If M is fixed, k must be a divisor of M (since k*sigma(k)=M implies k|M).
    # For each divisor k of M with k <= sqrt(M), check if k*sigma(k) = M.
    
    # Let's try to find M with many solutions by looking at M values
    # that have many divisors k where k*sigma(k) = M.
    
    # Alternative: For k up to some limit, compute k*sigma(k) and look for collisions.
    # We already did this above.
    
    # Let's also try to extend the search with a different approach:
    # For a given M, find all k such that k*sigma(k) = M.
    # k must divide M. For each divisor k of M with k <= sqrt(M), check.
    
    # But we don't know M in advance. Let's stick with the sieve approach.
    
    # Try to go higher if we have time
    remaining = deadline - time.time()
    if remaining > 20 and best_count < 4:
        # Try N = 10,000,000 but only if we have time
        N2 = 10000000
        sigma_arr2 = array('Q', [0]) * (N2 + 1)
        
        for d in range(1, N2 + 1):
            if time.time() > deadline - 5:
                break
            for m in range(d, N2 + 1, d):
                sigma_arr2[m] += d
        
        groups2 = defaultdict(list)
        for k in range(1, N2 + 1):
            if time.time() > deadline:
                break
            M_val = k * sigma_arr2[k]
            if M_val <= 10**18:
                groups2[M_val].append(k)
                if len(groups2[M_val]) > best_count:
                    best_count = len(groups2[M_val])
                    best_M = M_val
                    best_k = groups2[M_val][:]
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
