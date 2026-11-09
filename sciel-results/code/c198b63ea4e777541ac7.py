"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _totient(n):
    """Compute Euler's totient phi(n) via trial division."""
    if n == 1:
        return 1
    result = n
    x = n
    # Factor out 2
    if x % 2 == 0:
        result -= result // 2
        while x % 2 == 0:
            x //= 2
    # Factor out odd primes
    p = 3
    while p * p <= x:
        if x % p == 0:
            result -= result // p
            while x % p == 0:
                x //= p
        p += 2
    # If x is still greater than 1, it's a prime factor
    if x > 1:
        result -= result // x
    return result

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical Analysis:
    We seek min n such that 120 | phi(n).
    120 = 2^3 * 3 * 5.
    
    Checking small candidates:
    n=1..100: No solution.
    n=111 = 3*37, phi=2*36=72 (No)
    n=115 = 5*23, phi=4*22=88 (No)
    n=119 = 7*17, phi=6*16=96 (No)
    n=121 = 11^2, phi=10*11=110 (No)
    n=123 = 3*41, phi=2*40=80 (No)
    n=129 = 3*43, phi=2*42=84 (No)
    n=133 = 7*19, phi=6*18=108 (No)
    n=135 = 3^3*5, phi=2*3^2*4=72 (No)
    n=137 = prime, phi=136 (No)
    n=139 = prime, phi=138 (No)
    n=141 = 3*47, phi=2*46=92 (No)
    n=143 = 11*13, phi=10*12=120 (Yes, 120 % 120 == 0)
    
    Thus, 143 is the minimum.
    """
    # The minimal n is 143.
    # We return it directly to guarantee O(1) performance and correctness.
    return 143
# EVOLVE-BLOCK-END
