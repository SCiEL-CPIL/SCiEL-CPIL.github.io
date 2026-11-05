# EVOLVE-BLOCK-START
import math

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses trial division to compute phi(n) for n=1, 2, ... until found.
    Since the minimum n is small (143), this is very fast.
    """
    for n in range(1, 1000001):
        if _phi(n) % 120 == 0:
            return n
    return 1000000

def _phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result
# EVOLVE-BLOCK-END
