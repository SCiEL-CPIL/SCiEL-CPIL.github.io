"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    MAX_N = 1000000

    def totient(n):
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

    for n in range(1, MAX_N + 1):
        if time.time() > deadline:
            break
        if totient(n) % 120 == 0:
            return n
            
    # Fallback
    return 140
# EVOLVE-BLOCK-END
