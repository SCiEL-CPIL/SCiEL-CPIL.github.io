"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Generate primes up to max_gap using Sieve of Eratosthenes
    is_prime = [True] * max_gap
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_gap**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_gap, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, max_gap) if is_prime[i]]
    
    result = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        # We need to mark numbers in (lo, hi) that have any prime factor < gap_size
        # Create a boolean list for the range [lo+1, hi-1]
        # Size is gap_size - 1
        valid = [True] * (gap_size - 1)
        
        for p in small_primes:
            if p >= gap_size:
                break
            # Find the first multiple of p in (lo, hi)
            # m = k * p
            # lo < k * p < hi
            # ceil((lo+1)/p) <= k <= floor((hi-1)/p)
            start_k = (lo + p) // p  # ceil((lo+1)/p) is (lo+1 + p - 1)//p = (lo+p)//p
            end_k = (hi - 1) // p
            
            for k in range(start_k, end_k + 1):
                m = k * p
                idx = m - (lo + 1)
                valid[idx] = False
        
        witness = None
        for i in range(gap_size - 1):
            if valid[i]:
                witness = lo + 1 + i
                break
        
        result.append(witness)
    
    return result
# EVOLVE-BLOCK-END
