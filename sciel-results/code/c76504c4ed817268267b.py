import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for the 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max value of n is 1,000,000.
    # We look at n+1 to n+24.
    # Max index accessed is 1,000,000 + 24 = 1,000,024.
    MAX_IDX = 1000024
    MAX_N = 1000000
    MIN_N = 1000
    
    # Precompute smooth parts
    # smooth[i] will store the product of prime factors < 24 of i
    smooth = [1] * (MAX_IDX + 1)
    
    for p in primes:
        for mult in range(p, MAX_IDX + 1, p):
            val = mult
            while val % p == 0:
                smooth[mult] *= p
                val //= p
                
    # Sliding window
    # Window is smooth[n+1] ... smooth[n+24]
    # n ranges from MIN_N to MAX_N
    
    # Initialize window for n = MIN_N
    # Indices: MIN_N+1 to MIN_N+24
    window_counts = defaultdict(int)
    for i in range(MIN_N + 1, MIN_N + 25):
        window_counts[smooth[i]] += 1
        
    min_distinct = len(window_counts)
    best_n = MIN_N
    
    # Slide the window
    # Next n is MIN_N + 1
    # Remove smooth[MIN_N+1], Add smooth[MIN_N+25]
    for n in range(MIN_N + 1, MAX_N + 1):
        if time.time() > deadline:
            break
            
        # Remove the element leaving the window: index n
        # Previous window was [n-1+1 ... n-1+24] = [n ... n+23]
        # Current window is [n+1 ... n+24]
        # So we remove smooth[n] and add smooth[n+24]
        
        val_out = smooth[n]
        window_counts[val_out] -= 1
        if window_counts[val_out] == 0:
            del window_counts[val_out]
            
        val_in = smooth[n + 24]
        window_counts[val_in] = window_counts.get(val_in, 0) + 1
        
        current_distinct = len(window_counts)
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            if min_distinct == 1:
                # Can't get better than 1
                break
                
    return best_n
