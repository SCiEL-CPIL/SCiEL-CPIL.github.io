import math

def construct():
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    # No n < 143 has a totient divisible by 120.
    # We perform a quick verification via a small sieve to ensure correctness
    # within the design family, but the result is deterministic.
    
    limit = 200
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 143
