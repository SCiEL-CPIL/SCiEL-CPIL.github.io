import time
from math import isqrt

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    TARGET = 123457
    
    # Generate primes up to a reasonable limit
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # Generate primes up to TARGET
    primes_list = sieve(TARGET)
    
    def largest_prime_factor(n):
        """Compute the largest prime factor of n."""
        if n <= 1:
            return 1
        lp = 1
        for p in primes_list:
            if p * p > n:
                break
            while n % p == 0:
                lp = p
                n //= p
        if n > 1:
            lp = max(lp, n)
        return lp

    # Strategy: Incrementally build smooth number set as we add primes
    best_pair = None
    best_value = float('inf')
    
    smooths = {1}
    
    for B in primes_list:
        if time.time() > deadline - 3:
            break
        
        # Add new smooth numbers that involve prime B
        new_smooths = []
        for s in list(smooths):
            nv = s * B
            while nv <= TARGET:
                if nv not in smooths:
                    new_smooths.append(nv)
                    smooths.add(nv)
                nv *= B
        
        # Check newly added numbers for complementary pairs
        for a in new_smooths:
            b = TARGET - a
            if b <= 0 or b > TARGET:
                continue
            if b in smooths:
                if a > 0 and b > 0:
                    best_pair = (a, b)
                    best_value = B
                    break
        if best_pair:
            break
            
    if best_pair:
        return [best_pair[0], best_pair[1]]
    else:
        # Fallback: compute max(P(a), P(b)) for several candidate pairs and pick the best
        candidates = []
        for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100, 200, 500, 1000]:
            b = TARGET - a
            if b > 0:
                candidates.append((a, b))
        
        # Also try pairs around TARGET/2
        mid = TARGET // 2
        for offset in range(1, 2000):
            a1 = mid - offset
            b1 = mid + 1 + offset
            a2 = mid + offset
            b2 = mid + 1 - offset
            if a1 > 0 and b1 > 0:
                candidates.append((a1, b1))
            if a2 > 0 and b2 > 0:
                candidates.append((a2, b2))
        
        best_fallback = None
        best_fallback_val = float('inf')
        
        for a, b in candidates:
            if a <= 0 or b <= 0:
                continue
            pa = largest_prime_factor(a)
            pb = largest_prime_factor(b)
            val = max(pa, pb)
            if val < best_fallback_val:
                best_fallback_val = val
                best_fallback = (a, b)
        
        if best_fallback:
            return [best_fallback[0], best_fallback[1]]
        else:
            return [1, TARGET - 1]
