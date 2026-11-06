import time
import numpy as np

def construct():
    S = 123457
    deadline = time.time() + 80.0
    
    # 1. Compute Largest Prime Factor (Lpf) for all numbers up to S
    # Using a sieve approach: for each prime p, mark multiples
    # lpf[j] will hold the largest prime factor since we iterate primes in increasing order
    lpf = np.zeros(S + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Sieve: for each i, if lpf[i] == 0, i is prime
    # Then mark all multiples of i with lpf[j] = i
    # Since we go in increasing order, the last (largest) prime to divide j wins
    for i in range(2, S + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark all multiples of i
            multiples = np.arange(i, S + 1, i)
            lpf[multiples] = i
    
    if time.time() > deadline:
        return [1, S - 1]
    
    # 2. Bucket numbers by their Lpf value
    # Only keep buckets for Lpf values that actually occur
    buckets = {}
    for x in range(1, S + 1):
        p = int(lpf[x])
        if p not in buckets:
            buckets[p] = []
        buckets[p].append(x)
    
    if time.time() > deadline:
        return [1, S - 1]
    
    # 3. Iterate over distinct Lpf values in increasing order
    distinct_lpfs = sorted(buckets.keys())
    
    smooth_set = set()
    
    for k in distinct_lpfs:
        if time.time() > deadline:
            return [1, S - 1]
        
        new_numbers = buckets[k]
        
        # Check: for each new number x, is S-x already in smooth_set?
        for x in new_numbers:
            comp = S - x
            if 1 <= comp <= S and comp in smooth_set:
                return [x, comp]
        
        # Check: do any two new numbers sum to S?
        if len(new_numbers) > 1:
            new_set = set(new_numbers)
            for x in new_numbers:
                comp = S - x
                if comp in new_set:
                    return [x, comp]
        
        # Add new numbers to smooth_set
        for x in new_numbers:
            smooth_set.add(x)
    
    # Fallback
    return [1, S - 1]
