import time
import math

def construct():
    S = 123457
    deadline = time.time() + 50
    
    # Step 1: Compute largest prime factor for all numbers up to S
    # lpf[n] = largest prime factor of n, lpf[1] = 1
    lpf = [0] * (S + 1)
    lpf[1] = 1
    for i in range(2, S + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, S + 1, i):
                lpf[j] = i
    if time.time() > deadline:
        return [1, S - 1]
    
    # Step 2: Group numbers by their lpf value
    # numbers_by_lpf[p] = list of numbers x in [1, S] with lpf[x] == p
    numbers_by_lpf = {}
    for x in range(1, S + 1):
        p = lpf[x]
        if p not in numbers_by_lpf:
            numbers_by_lpf[p] = []
        numbers_by_lpf[p].append(x)
    
    if time.time() > deadline:
        return [1, S - 1]
    
    # We want to find the smallest K such that there exists x in [1, S] with
    # x + (S-x) = S, lpf[x] <= K, and lpf[S-x] <= K.
    
    # Strategy: iterate K from 1 upward. Maintain a set `smooth_set` of all x
    # in [1, S] with lpf[x] <= K. At each K, add numbers whose lpf == K.
    # Then check if any x in the newly added numbers has its complement S-x
    # also in smooth_set.
    
    smooth_set = set()
    
    # Get sorted list of distinct lpf values that actually occur
    distinct_lpfs = sorted(numbers_by_lpf.keys())
    
    for k in distinct_lpfs:
        if time.time() > deadline:
            return [1, S - 1]
        
        new_numbers = numbers_by_lpf[k]
        
        # Check: for each new number x, is S-x already in smooth_set?
        for x in new_numbers:
            comp = S - x
            if 1 <= comp <= S and comp in smooth_set:
                return [x, comp]
        
        # Check: do any two new numbers sum to S?
        # i.e., x1 + x2 = S with x1, x2 both in new_numbers
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
