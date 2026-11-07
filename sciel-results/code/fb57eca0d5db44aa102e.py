"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    # Precompute primes up to a reasonable bound (say 200) to cover any gap
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(200)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Determine relevant primes: those strictly less than gap
        relevant_primes = [p for p in primes if p < gap]
        
        witness = None
        
        # First try CRT approach: find m = k*M + 1 in (lo, hi)
        # where M is the product of all primes < gap
        if relevant_primes:
            M = 1
            for p in relevant_primes:
                M *= p
            
            # We want lo < k*M + 1 < hi
            # => (lo - 1) / M < k < (hi - 1) / M
            # => k_min = floor((lo - 1) / M) + 1
            # => k_max = ceil((hi - 1) / M) - 1 = (hi - 2) // M
            
            k_min = (lo - 1) // M + 1
            k_max = (hi - 2) // M
            
            if k_min <= k_max:
                # Try k values from k_min to k_max
                for k in range(k_min, k_max + 1):
                    m = k * M + 1
                    # m is guaranteed to be > lo and < hi by construction
                    # Verify m is strictly between lo and hi
                    if lo < m < hi:
                        # Verify m is not divisible by any prime < gap
                        # (This should be true by CRT, but let's double-check)
                        valid = True
                        for p in relevant_primes:
                            if m % p == 0:
                                valid = False
                                break
                        if valid:
                            witness = m
                            break
        
        # If CRT didn't find a witness, fall back to sequential search
        if witness is None:
            for m in range(lo + 1, hi):
                # Check if m is divisible by any relevant prime
                is_valid = True
                for p in relevant_primes:
                    if m % p == 0:
                        is_valid = False
                        break
                
                if is_valid:
                    witness = m
                    break
        
        result[i] = int(witness) if witness is not None else None
    
    return result

# EVOLVE-BLOCK-END
