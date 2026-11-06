"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: square-free n with the maximum number of distinct prime factors.
The product of the first k primes, P_k, is the smallest square-free integer
with k distinct primes. We choose the largest k with P_k <= 10^18.

P_15 = 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
P_16 = P_15 * 53 > 10^18

Square-free numbers maximize divisor count per unit of n (2^k divisors for
k primes), and the hypercube divisor lattice has dense coprime-adjacent
transitions because sorted divisors frequently differ by a single prime
factor, creating adjacent pairs with disjoint prime supports.
"""
# EVOLVE-BLOCK-START
def construct():
    # Product of the first 15 primes: 614889782588491410
    # This is the largest square-free product of consecutive primes <= 10^18
    return 614889782588491410
# EVOLVE-BLOCK-END
