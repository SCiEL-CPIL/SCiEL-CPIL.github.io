"""Maximize coprime adjacent divisor pairs using primorial with max distinct primes."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Product of first 15 primes (47#) = 614889782588491410 <= 10^18
    # This maximizes the number of distinct prime factors, giving 2^15 = 32768 divisors
    # More divisors means more opportunities for coprime adjacent pairs
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
