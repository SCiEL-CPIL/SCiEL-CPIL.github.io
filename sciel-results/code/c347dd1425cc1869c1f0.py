"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80
    
    def phi(n):
        """Compute Euler's totient function for n."""
        result = n
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
            p += 1
        if m > 1:
            result -= result // m
        return result
    
    # Search for smallest n with phi(n) divisible by 120
    for n in range(1, 1000001):
        if time.time() > deadline:
            break
        if phi(n) % 120 == 0:
            return n
    
    # Fallback
    return 14400
# EVOLVE-BLOCK-END
