import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80
    
    SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW
    
    # Precompute smooth part using sieve
    smooth = list(range(MAX_VAL + 1))  # start with identity
    # For each small prime p, for multiples of p, divide out all factors of p
    for p in SMALL_PRIMES:
        for multiple in range(p, MAX_VAL + 1, p):
            x = multiple
            while x % p == 0:
                x //= p
            smooth[multiple] = multiple // x  # this is the p-part times previous... 
    
    # Hmm, that's not quite right. Let me think again.
    # smooth[m] should be the product of p^a for each small prime p where p^a || m
    # Better: smooth[m] = m / rough[m] where rough[m] has no small prime factors
    
    # Alternative: compute rough part, then smooth = m // rough
    rough = list(range(MAX_VAL + 1))
    for p in SMALL_PRIMES:
        for multiple in range(p, MAX_VAL + 1, p):
            while rough[multiple] % p == 0:
                rough[multiple] //= p
    
    # Now smooth[m] = m // rough[m]
    # But we don't need to store both, just compute on the fly or store smooth
    
    # Actually let's just store smooth directly
    smooth = [1] * (MAX_VAL + 1)
    for m in range(2, MAX_VAL + 1):
        x = m
        sp = 1
        for p in SMALL_PRIMES:
            while x % p == 0:
                sp *= p
                x //= p
        smooth[m] = sp
    
    # Sliding window
    freq = defaultdict(int)
    for i in range(1001, 1001 + WINDOW):
        freq[smooth[i]] += 1
    
    best_n = 1000
    best_distinct = len(freq)
    
    for n in range(1000, MAX_N + 1):
        if time.time() > deadline:
            break
        
        current_distinct = len(freq)
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = n
            if best_distinct <= 2:
                break
        
        if n + 25 <= MAX_VAL:
            out_val = smooth[n + 1]
            freq[out_val] -= 1
            if freq[out_val] == 0:
                del freq[out_val]
            
            in_val = smooth[n + 25]
            freq[in_val] += 1
    
    return best_n
