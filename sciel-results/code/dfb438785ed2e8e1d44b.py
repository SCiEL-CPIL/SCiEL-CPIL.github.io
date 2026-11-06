import time
import math

def construct():
    """
    Return smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Precompute smallest prime factor (SPF) up to 1000000
    N = 1000000
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def phi_fast(n):
        """Compute phi(n) using precomputed SPF."""
        if n == 1:
            return 1
        result = n
        while n > 1:
            p = spf[n]
            result -= result // p
            while n % p == 0:
                n //= p
        return result

    # Search for smallest n with phi(n) % 120 == 0
    for n in range(1, N + 1):
        if time.time() - start_time > deadline:
            break
        if phi_fast(n) % 120 == 0:
            return n

    return 1000000  # fallback
