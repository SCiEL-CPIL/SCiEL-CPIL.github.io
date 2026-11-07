"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

# Precompute primes up to max possible gap
_max_gap = max(hi - lo for lo, hi in P['gaps'])
_primes = _sieve(_max_gap + 1)

def _lp_ge(m, threshold, primes):
    """Check if least prime factor of m is >= threshold.
    i.e., m is not divisible by any prime < threshold."""
    for p in primes:
        if p >= threshold:
            return True
        if m % p == 0:
            return False
    return True

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    # Pre-compute relevant primes for each gap to avoid repeated list filtering
    # relevant_primes[i] will be the list of primes < gap_i
    gap_primes_list = []
    for lo, hi in P['gaps']:
        gap = hi - lo
        # Primes strictly less than gap
        rp = [p for p in _primes if p < gap]
        gap_primes_list.append(rp)

    for i, (lo, hi) in enumerate(P['gaps']):
        # Check deadline
        if time.time() - _start_time > _DEADLINE:
            break
        
        gap = hi - lo
        relevant_primes = gap_primes_list[i]
        
        # Exhaustive forward search
        for m in range(lo + 1, hi):
            if _lp_ge(m, gap, relevant_primes):
                result[i] = m
                break
    
    # Ensure all entries are Python int or None
    # The values are already Python ints from range()
    return result

# EVOLVE-BLOCK-END
