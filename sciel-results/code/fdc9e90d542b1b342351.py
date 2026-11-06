"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Square-free n with the maximum number of distinct prime factors.
We use the product of the first 15 primes (P_15) which gives 2^15 = 32768 divisors.
The hypercube structure of the divisor lattice for square-free numbers creates
many adjacent coprime pairs in the sorted divisor list.

P_15 = 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
P_16 = P_15 * 53 > 10^18
"""
# EVOLVE-BLOCK-START
def construct():
    # Product of the first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
