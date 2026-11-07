"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return list of primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = 0
    return [i for i in range(2, n + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Max gap in our data is 16 (10009->10037 is 28, let me check)
    # Actually: 10009 to 10037 = 28, 10039 to 10061 = 22, 10273 to 10289 = 16
    # Let me compute properly
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need primes less than max_gap to check divisibility
    # For the condition "least prime factor >= gap", we check if any prime < gap divides m
    primes_small = _sieve(max_gap + 1)
    
    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        
        # Primes to check: all primes strictly less than gap
        check_primes = [p for p in primes_small if p < gap]
        
        found = False
        if len(check_primes) == 0:
            # No primes to check, any integer in (lo, hi) works
            if hi - lo > 1:
                result[i] = int(lo + 1)
                found = True
        else:
            # Strategy: scan from lo+1 upward
            for m in range(lo + 1, hi):
                ok = True
                for p in check_primes:
                    if m % p == 0:
                        ok = False
                        break
                if ok:
                    result[i] = int(m)
                    found = True
                    break
            
            # Strategy 2: if not found, scan from hi-1 downward
            if not found:
                for m in range(hi - 1, lo, -1):
                    ok = True
                    for p in check_primes:
                        if m % p == 0:
                            ok = False
                            break
                    if ok:
                        result[i] = int(m)
                        found = True
                        break
                        
            # Strategy 3: if still not found, try middle out
            if not found:
                mid = (lo + hi) // 2
                for offset in range(0, (hi - lo) // 2 + 1):
                    for m in [mid + offset, mid - offset]:
                        if m <= lo or m >= hi:
                            continue
                        ok = True
                        for p in check_primes:
                            if m % p == 0:
                                ok = False
                                break
                        if ok:
                            result[i] = int(m)
                            found = True
                            break
                    if found:
                        break
    
    # Ensure types are correct: integers must be Python int
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
