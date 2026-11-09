"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    """
    Finds an integer witness w for each prime gap (p, q) such that p < w < q
    and the least prime factor of w is at least the gap size (q - p).
    Returns a list of 32 entries, each an int witness or None.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    if n_gaps == 0:
        return result

    # 1. Determine the maximum value needed for the sieve
    max_prime = max(hi for _, hi in gaps)
    
    # 2. Compute Least Prime Factor (LPF) for all numbers up to max_prime
    # lpf[n] will store the smallest prime that divides n.
    # If n is prime, lpf[n] = n.
    lpf = list(range(max_prime + 1))
    
    # Sieve for LPF
    for i in range(2, isqrt(max_prime) + 1):
        if lpf[i] == i: # i is prime
            for j in range(i * i, max_prime + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    # 3. Iterate through each gap and find a witness
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
            
        gap_size = hi - lo
        
        # We need w such that lo < w < hi and lpf[w] >= gap_size
        # Note: If w is prime, lpf[w] = w. Since w < hi <= ~10300, 
        # and gap_size is small (<= 20), primes are always valid witnesses 
        # provided w >= gap_size. 
        # However, composite numbers can also be valid if their smallest 
        # factor is large enough.
        
        found_witness = None
        for w in range(lo + 1, hi):
            # Check condition: least prime factor of w is at least gap_size
            if lpf[w] >= gap_size:
                found_witness = w
                break
        
        if found_witness is not None:
            result[i] = int(found_witness)
        else:
            result[i] = None

    return result

# EVOLVE-BLOCK-END
