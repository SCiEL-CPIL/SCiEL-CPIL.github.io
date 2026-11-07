import math
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_below(n):
    """Return sorted list of primes strictly less than n."""
    if n <= 2:
        return []
    sieve = bytearray(b'\x01') * n
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(math.isqrt(n - 1)) + 1):
        if sieve[i]:
            sieve[i*i:n:i] = bytearray(len(range(i*i, n, i)))
    return [i for i in range(2, n) if sieve[i]]

def construct():
    t0 = time.monotonic()
    hard_deadline = t0 + 50.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    if not gaps:
        return result

    max_gap = max(hi - lo for lo, hi in gaps)
    small_primes = _primes_below(max_gap + 1) # primes < max_gap+1, i.e., <= max_gap. We need p < g.
    
    # Precompute the product of forbidden primes for each unique gap size
    # If product is 1 (empty set), then gcd(m, 1) is always 1.
    forbidden_product_by_gap = {}
    for g in set(hi - lo for lo, hi in gaps):
        prod = 1
        for p in small_primes:
            if p < g:
                prod *= p
        forbidden_product_by_gap[g] = prod

    for i in range(n):
        if time.monotonic() > hard_deadline:
            break
        lo, hi = gaps[i]
        g = hi - lo
        prod = forbidden_product_by_gap[g]
        
        found = False
        # Try from left
        for m in range(lo + 1, hi):
            if math.gcd(m, prod) == 1:
                result[i] = int(m)
                found = True
                break
        
        if not found:
            # Try from right
            for m in range(hi - 1, lo, -1):
                if math.gcd(m, prod) == 1:
                    result[i] = int(m)
                    found = True
                    break

        if not found:
            result[i] = None

    return result
