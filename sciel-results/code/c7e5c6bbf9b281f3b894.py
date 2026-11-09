"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
_start = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def construct():
    # Deadline guard
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    n = 256
    
    # Sieve to find all primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    # Strategy 1: Max Prime Powers
    set1 = [1]
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        set1.append(val)
    sum1 = sum(set1)
    
    # Strategy 2: Greedy Merge
    current_set = list(primes) + [1]
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_pair = None
        
        s = sorted(current_set)
        for i in range(len(s)):
            if s[i] > 256:
                continue
            for j in range(i + 1, len(s)):
                prod = s[i] * s[j]
                if prod > 256:
                    break
                if prod in current_set:
                    continue
                gain = prod - s[i] - s[j]
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (s[i], s[j])
        
        if best_pair:
            a, b = best_pair
            current_set.remove(a)
            current_set.remove(b)
            current_set.append(a * b)
            improved = True
            
        if time.time() - _start > _TIMEOUT - 5:
            break
            
    sum2 = sum(current_set)
    
    # Verify and return the better set
    from math import gcd
    
    def is_valid(s):
        if len(s) != len(set(s)):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    if sum1 >= sum2:
        result = set1
    else:
        result = current_set
        
    if not is_valid(result):
        # Fallback to primes if something went wrong
        return [1] + primes

    return [int(x) for x in result]
# EVOLVE-BLOCK-END
