"""Rewritten program: Vectorized Segmented Sieve for witness search."""
# EVOLVE-BLOCK-START
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_upto(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    gaps = P['gaps']
    result = [None] * len(gaps)
    
    # Precompute primes up to a safe limit (max gap is ~28, so 50 is plenty)
    max_gap = max(hi - lo for lo, hi in gaps)
    all_primes = _primes_upto(max_gap + 10)
    
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        # Candidates are lo+1 ... hi-1
        num_candidates = hi - lo - 1
        if num_candidates <= 0:
            continue
            
        # Primes strictly less than gap
        small_primes = [p for p in all_primes if p < gap]
        
        # Create array of candidate values
        # Using int64 to be safe, though int32 is enough
        candidates = np.arange(lo + 1, hi, dtype=np.int64)
        
        # Initialize mask to True (valid)
        valid = np.ones(num_candidates, dtype=bool)
        
        # Sieve out multiples of small primes
        for p in small_primes:
            # Find the first multiple of p in the range [lo+1, hi-1]
            # first_multiple = ceil((lo+1)/p) * p
            first_multiple = ((lo + 1 + p - 1) // p) * p
            
            if first_multiple < hi:
                # Indices in the candidates array corresponding to multiples of p
                # candidates[j] = lo + 1 + j
                # We want candidates[j] % p == 0
                # So lo + 1 + j = k * p
                # j = k * p - lo - 1
                # The first k is such that k*p >= lo+1
                # k_min = ceil((lo+1)/p)
                # j_min = k_min * p - lo - 1
                
                k_min = (lo + 1 + p - 1) // p
                j_min = k_min * p - (lo + 1)
                
                if j_min < num_candidates:
                    # Mark all indices j_min, j_min+p, j_min+2p, ... as invalid
                    valid[j_min::p] = False
        
        # Check if any valid candidate remains
        if np.any(valid):
            # Get the index of the first True
            first_idx = np.argmax(valid)
            witness = int(candidates[first_idx])
            result[i] = witness
        else:
            result[i] = None
            
    return result
# EVOLVE-BLOCK-END
