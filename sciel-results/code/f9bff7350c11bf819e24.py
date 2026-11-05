import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute smooth parts
    extracted = [1] * (MAX_VAL + 1)
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # Sliding window distinct count
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # We need to scan n from N_MIN to N_MAX
    # Window for n is extracted[n+1 : n+1+WINDOW]
    
    # Initialize frequency for first window (n = N_MIN)
    freq = {}
    current_distinct = 0
    
    # Helper to update freq and distinct count
    # We will just maintain a dict and count keys
    
    # First window: n = N_MIN
    # Elements: extracted[N_MIN+1] ... extracted[N_MIN+WINDOW]
    for i in range(1, WINDOW + 1):
        val = extracted[N_MIN + i]
        if val not in freq:
            freq[val] = 0
            current_distinct += 1
        freq[val] += 1
        
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = N_MIN

    # Slide window
    # Next n is N_MIN + 1
    # Remove extracted[N_MIN+1], Add extracted[N_MIN+1+WINDOW]
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove element at n (which is extracted[n])
        # Wait, window for n-1 was [n, n+WINDOW-1]
        # Window for n is [n+1, n+WINDOW]
        # So remove extracted[n], add extracted[n+WINDOW]
        
        val_out = extracted[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            current_distinct -= 1
            
        val_in = extracted[n + WINDOW]
        if val_in not in freq:
            freq[val_in] = 0
            current_distinct += 1
        freq[val_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break

    return best_n
