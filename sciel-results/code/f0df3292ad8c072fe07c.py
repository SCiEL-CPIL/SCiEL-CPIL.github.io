# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill
    
    # Generate primes up to max gap size (max gap is 28: 10273->10289)
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes up to max_gap to check divisibility
    small_primes = _gen_primes(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find smallest prime >= gap
        # We need m such that least_prime_factor(m) >= gap
        # This means m is not divisible by any prime < gap
        
        # Generate primes less than gap that we need to avoid
        # Primes to avoid: all primes p where p < gap
        avoid_primes = [p for p in small_primes if p < gap]
        
        # Search for m in (lo, hi) not divisible by any avoid_prime
        found = False
        for m in range(lo + 1, hi):
            valid = True
            for p in avoid_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        if not found:
            # Try from the other end
            for m in range(hi - 1, lo, -1):
                valid = True
                for p in avoid_primes:
                    if m % p == 0:
                        valid = False
                        break
                if valid:
                    result[i] = m
                    break
    
    # Ensure all entries are Python int or None
    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))
    
    return final_result
# EVOLVE-BLOCK-END
