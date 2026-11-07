"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    result = [None] * len(P['gaps'])
    # Precompute small primes for factor checking
    # Max gap is 20, so we need primes up to 19
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # 1. Try to find a prime in (lo, hi)
        # A prime m in (lo, hi) has lp = m.
        # We need lp >= gap. Since m > lo >= 10000 and gap <= 20, m >= gap is always true.
        found_prime = False
        for m in range(lo + 1, hi):
            if _is_prime(m):
                result[i] = m
                found_prime = True
                break
        
        if found_prime:
            continue
            
        # 2. If no prime, find a composite m in (lo, hi) such that lp(m) >= gap.
        # This means m is not divisible by any prime p < gap.
        # Let small_factors be primes < gap.
        small_factors = [p for p in small_primes if p < gap]
        
        for m in range(lo + 1, hi):
            # Skip if m is prime (already checked, but safe)
            if _is_prime(m):
                # Should not happen if step 1 failed, but if it does, it's valid
                result[i] = m
                break
            # Check if m has any factor in small_factors
            has_small_factor = False
            for p in small_factors:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                # Least prime factor is >= gap
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
