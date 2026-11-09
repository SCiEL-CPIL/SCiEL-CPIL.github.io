import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Generate primes up to 2000
    limit = 2000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            lpf = n
        return lpf
    
    best_n = 10000
    lpf_curr = largest_prime_factor(10000)  # lpf of current n
    lpf_next = largest_prime_factor(10001)  # lpf of n+1
    best_val = max(lpf_curr, lpf_next)
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        
        # For candidate n: lpf(n) = lpf_next (from previous iteration's n+1)
        # Wait no. Let me re-think.
        
        # Previous iteration ended with: lpf_curr = lpf(n-1), lpf_next = lpf(n)
        # So for current candidate n: lpf(n) = lpf_next, need lpf(n+1)
        
        lpf_next = largest_prime_factor(n + 1)
        val = max(lpf_next, lpf_next)  # WRONG
        
        # Let me just be explicit:
        # At the start of this iteration for candidate n:
        #   lpf_curr holds lpf(n) [computed as lpf_next in previous iteration]
        #   We need lpf(n+1)
        
        lpf_n = lpf_curr
        lpf_np1 = largest_prime_factor(n + 1)
        val = max(lpf_n, lpf_np1)
        if val < best_val:
            best_val = val
            best_n = n
        
        lpf_curr = lpf_np1
    
    return best_n
