"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin before 90s kill

    # Precompute smallest prime factor for numbers up to max candidate + 1
    max_candidate = max(hi for lo, hi in P['gaps'])
    spf = list(range(max_candidate + 1))
    for i in range(2, int(max_candidate**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def least_prime_factor(m):
        """Return the least prime factor of m using precomputed SPF table."""
        if m <= max_candidate:
            return spf[m]
        # Fallback for larger m (shouldn't happen given our range)
        from math import isqrt
        for d in range(2, isqrt(m) + 1):
            if m % d == 0:
                return d
        return m

    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap
        # Check all interior candidates; gaps are small so this is fast
        # Prioritize candidates with higher lpf for robustness
        best_m = None
        best_lpf = 0
        for m in range(lo + 1, hi):
            lpf = least_prime_factor(m)
            if lpf >= gap:
                # Found a valid witness
                result[i] = int(m)
                break
            elif lpf > best_lpf:
                best_lpf = lpf
                best_m = m
    return result
# EVOLVE-BLOCK-END
