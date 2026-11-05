import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # seconds, safety margin before 60s limit

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m] = product of p^k for p<24 where p^k || m
    # Using sieve: for each prime p, for each power p^k, multiply all multiples of p^k by p
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # We want to minimize the number of distinct values in extracted[n+1 : n+25]
    # Iterate n from N_MIN to N_MAX
    
    # Optimization: 
    # The window size is 24. 
    # We can maintain a frequency dictionary or simply use a set for the current window.
    # Since we slide by 1, we can update the set.
    # However, set membership and update in Python is fast enough for 10^6 iterations?
    # 10^6 iterations * ~1us per op = ~1s. Should be fine.
    
    # Let's use a list for the current window values and count unique.
    # Actually, maintaining a count of distinct items in a sliding window is O(1) amortized 
    # if we use a frequency map.
    
    from collections import defaultdict
    freq = defaultdict(int)
    
    # Initialize window for n = N_MIN
    # Values are extracted[N_MIN+1] ... extracted[N_MIN+24]
    for i in range(1, WINDOW + 1):
        freq[extracted[N_MIN + i]] += 1
        
    distinct_count = len(freq)
    best_n = N_MIN
    best_count = distinct_count
    
    # If we already found a very low count, we can break early?
    # The theoretical minimum is likely 5 or 6. 
    # If we find 5, we can probably stop or continue to see if 4 is possible (unlikely).
    # Let's aim for the global minimum.
    
    # To speed up, we can check time every 10000 steps.
    for n in range(N_MIN + 1, N_MAX + 1):
        if n % 10000 == 0 and time.time() - start_time > DEADLINE:
            break

        # Remove extracted[n] from window
        # Window for n-1: [n-1+1, n-1+24] = [n, n+23]
        # Window for n: [n+1, n+24]
        # Remove: extracted[n]
        # Add: extracted[n+24]
        
        old_val = extracted[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1
            
        new_val = extracted[n + 24]
        if new_val not in freq:
            freq[new_val] = 1
            distinct_count += 1
        else:
            freq[new_val] += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            # If we find a count of 5, it's a very strong candidate. 
            # Can we go lower? 
            # With 24 numbers, and primes 2,3,5,7,11,13,17,19,23.
            # It is highly unlikely to have fewer than 5 distinct types.
            # If best_count == 5, we can continue to see if 4 is possible, but 5 is likely the floor.
            # Let's break if we find 5 to save time? No, let's just keep the best.
            # Actually, if we find 5, we can probably stop searching for 4 immediately as it's extremely rare.
            # But let's just run the full search if time permits.

    return best_n
