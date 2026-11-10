def construct():
    # Maximize tau_perp(n): number of adjacent coprime pairs in sorted divisors of n.
    # For squarefree n with k distinct primes, divisors are 2^k subset products.
    # More distinct primes => more divisors => more adjacent pairs => higher potential score.
    # Max k for n <= 10^18 is 15 (product of first 15 primes ≈ 6.15e17; adding 53 exceeds 10^18).
    # The first 15 primes (smallest) give the densest divisor lattice, maximizing coprime adjacencies.
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    n = 1
    for p in primes:
        n *= p
    return n
