import math

def construct():
    N = 123457
    # Sieve for largest prime factor
    lpf = [1] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard sieve to identify primes
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
                
    # Fill largest prime factor
    # Iterate p from 2 to N. If p is prime, it is the largest prime factor for its multiples
    # that haven't been updated by a larger prime? No.
    # If we iterate p in increasing order, and set lpf[m] = p for all multiples m of p,
    # the last prime p that divides m will be the largest prime factor.
    for p in range(2, N + 1):
        if is_prime[p]:
            for m in range(p, N + 1, p):
                lpf[m] = p
                
    best_max_p = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            if best_max_p == 1:
                break
            if best_max_p == 2:
                # Only possible if a=1 or b=1 and other is power of 2?
                # N is odd, so one is even, one is odd.
                # If max is 2, one must be power of 2, other must be 1?
                # P(1)=1. P(2^k)=2.
                # a=1, b=N-1. P(N-1) must be 2. N-1 is power of 2.
                # N=123457. N-1=123456. Not power of 2.
                # So max 2 is impossible for this N.
                # But keeping the check is harmless.
                pass

    return [best_a, best_b]
