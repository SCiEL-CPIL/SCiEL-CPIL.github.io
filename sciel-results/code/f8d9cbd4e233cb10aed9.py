import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

# Precompute primes up to a sufficient bound for least prime factor computation
MAX_VAL = max(hi for _, hi in P['gaps'])
# We need primes up to sqrt(MAX_VAL) for LPF, but actually for LPF we need primes up to MAX_VAL
# However, for checking if LPF[m] >= gap, we only need to check divisibility by primes < gap
# Since max gap is 20, we need primes < 20, which is just [2,3,5,7,11,13,17,19]
# But to be safe and general, let's compute LPF properly using a segmented approach

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Get the range we need to cover
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    
    # We need LPF for all numbers in (min_val, max_val)
    # Use a segmented sieve approach to compute LPF for the range [min_val, max_val]
    
    # First, get all primes up to sqrt(max_val)
    limit = int(max_val ** 0.5) + 1
    small_primes = sieve(limit)
    
    # Initialize LPF array for the range
    # We'll use a dictionary or array indexed by value
    # Since the range is small (~280 numbers), we can just use a list
    
    # Actually, let's just compute LPF for each number in the range directly
    # For each m in (lo, hi), check divisibility by primes < gap
    
    # Precompute relevant primes for each possible gap
    MAX_GAP = max(hi - lo for lo, hi in P['gaps'])
    # Get primes up to MAX_GAP
    all_primes = sieve(MAX_GAP)
    RELEVANT_PRIMES = {}
    for g in range(2, MAX_GAP + 1):
        RELEVANT_PRIMES[g] = [p for p in all_primes if p < g]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Special case: gap=2, any m in (lo, hi) works since LPF[m] >= 2 for any m > 1
        if gap == 2:
            result[i] = lo + 1
            continue
        
        relevant = RELEVANT_PRIMES[gap]
        
        # Search for witness: m strictly between lo and hi with LPF[m] >= gap
        # This means m is not divisible by any prime < gap
        for m in range(lo + 1, hi):
            is_valid = True
            for p in relevant:
                if m % p == 0:
                    is_valid = False
                    break
            
            if is_valid:
                result[i] = m
                break
    
    # Ensure all values are Python int, not numpy types
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
