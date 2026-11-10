"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

def phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n = n // p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

# EVOLVE-BLOCK-START
def construct():
    # We know 143 is a candidate. 
    # 143 = 11 * 13. phi(143) = 10 * 12 = 120.
    # We can just return 143.
    # To be robust, we can search up to 200.
    for n in range(1, 200):
        if phi(n) % 120 == 0:
            return n
    return 143
# EVOLVE-BLOCK-END
