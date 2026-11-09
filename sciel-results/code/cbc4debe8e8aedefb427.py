import time
import sys

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MAX = 1000000
    MAX_VAL = N_MAX + 24
    
    # Precompute small_part for all numbers up to MAX_VAL
    # small_part[x] = the largest divisor of x composed only of primes in 'primes'
    small_part = [1] * (MAX_VAL + 1)
    
    # For each prime, extract its full power from each multiple
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            val = multiple
            while val % p == 0:
                small_part[multiple] *= p
                val //= p
    
    # Now search for n that minimizes distinct small parts in window [n+1, n+24]
    # Use a sliding window with a frequency map
    
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: indices 1001 to 1024
    freq = {}
    for i in range(1, 25):
        sp = small_part[1000 + i]
        freq[sp] = freq.get(sp, 0) + 1
    distinct = len(freq)
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # If we already found a very good answer, we can potentially stop early
    # but let's continue searching
    
    # Slide the window: from n to n+1
    # Window for n: {n+1, ..., n+24}
    # Window for n+1: {n+2, ..., n+25}
    # Remove: n+1, Add: n+25
    
    for n in range(1001, N_MAX + 1):
        if time.time() > deadline:
            break
        
        remove_val = small_part[n + 1]
        add_val = small_part[n + 25]
        
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            del freq[remove_val]
        
        freq[add_val] = freq.get(add_val, 0) + 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            # If we found 1 or 2, that's excellent; keep searching for 1
            if best_count <= 1:
                # Can't do better than 1, but let's see if we can confirm
                pass
    
    return best_n
