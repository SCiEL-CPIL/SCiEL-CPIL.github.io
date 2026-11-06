"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    MAX_M = 128
    
    moduli = list(range(2, MAX_M + 1))
    
    primes = [m for m in moduli if _is_prime(m)]
    composites = [m for m in moduli if not _is_prime(m)]
    
    selected = []
    
    # Primes are coprime to all other moduli (except themselves, which are distinct)
    # So we can assign residue 0 to all of them safely.
    for m in primes:
        selected.append([0, m])
        
    # Sort composites by ascending value
    # Small moduli have fewer residue options, so assign them first
    composites.sort()
    
    for m in composites:
        found = False
        for r in range(m):
            valid = True
            for r_j, m_j in selected:
                g = _gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                found = True
                break
        if not found:
            pass
            
    return selected
# EVOLVE-BLOCK-END
