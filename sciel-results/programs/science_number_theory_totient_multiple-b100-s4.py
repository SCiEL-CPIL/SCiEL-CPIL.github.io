"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0

    def totient(n):
        result = n
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
            p += 1 if p == 2 else 2
        if m > 1:
            result -= result // m
        return result

    # The smallest n with phi(n) divisible by 120 is 143 (phi=120).
    # We search up to 1000 to be safe and fast.
    for n in range(1, 1001):
        if time.time() > deadline:
            break
        if totient(n) % 120 == 0:
            return n
            
    # Fallback if not found in small range (unlikely)
    # Search up to 10000
    for n in range(1001, 10001):
        if time.time() > deadline:
            break
        if totient(n) % 120 == 0:
            return n

    # Known valid candidate: 30030
    return 30030
# EVOLVE-BLOCK-END
