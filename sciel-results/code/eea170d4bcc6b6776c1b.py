"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill

    MAX_N = 256

    # Generate all primes up to MAX_N using Sieve of Eratosthenes
    def get_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    primes = get_primes(MAX_N)

    # We can always include 1 (gcd(1, x) = 1 for all x)
    # The problem reduces to: select a subset of numbers from 2..256 such that
    # any two selected numbers are coprime, maximizing the sum.
    #
    # Key insight: if we select a number n > 1, we cannot select any other number
    # sharing a prime factor with n. So each prime p can "own" at most one selected number.
    # The best number to assign to prime p is the largest power of p within 1..256.
    # But we can also assign composite numbers if they use multiple primes' "slots" - 
    # actually no, a composite number uses multiple primes simultaneously, which is wasteful
    # unless it's larger than the sum of individual prime powers.
    #
    # Actually, let's think more carefully. Each selected number n >= 2 has a set of prime factors.
    # Two numbers are coprime iff their prime factor sets are disjoint.
    # So we're selecting a collection of integers where the prime factor sets are pairwise disjoint.
    #
    # This is equivalent to: partition some primes into groups, where each group corresponds to
    # one selected number (the product of primes in that group, possibly with exponents).
    #
    # For maximum sum, for each prime p, the best use is either:
    # - Use p^k (largest power <= 256) as its own number
    # - Combine with other primes into a product
    #
    # Since combining primes into one number means we lose the ability to use them separately,
    # and a product a*b is typically less than a + b for a,b >= 2 (except 2*2=4 > 2+2=4, equal),
    # it's generally better to keep primes separate as their highest powers.
    #
    # Let's verify: for prime p, largest power p^k <= 256.
    # Compare keeping p^k alone vs combining with q^m: p^k * q^m vs p^k + q^m.
    # p^k * q^m > p^k + q^m iff (p^k-1)(q^m-1) > 1, which is true for p^k >= 2, q^m >= 2
    # except when p^k=2, q^m=2: 4 vs 4, equal.
    # So combining is never better or equal except the 2*2=4 case.
    #
    # Therefore, optimal strategy: for each prime, take the largest power <= 256, and include 1.
    
    result = [1]
    for p in primes:
        # Find largest power of p <= MAX_N
        power = p
        while power * p <= MAX_N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality (should be guaranteed by construction)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # This should never happen, but fallback to just primes
                result = [1] + primes
                break
    
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
