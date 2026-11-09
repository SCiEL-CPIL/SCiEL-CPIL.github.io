"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 82  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # We need to evaluate n in [1000, 1000000], so we look at n+i for i in 1..24
    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute sp[m] = complete factor of m composed of primes < 24
    # Initialize with 1s
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)

    # For each prime p < 24, multiply sp by p^k for all m divisible by p^k
    for p in PRIMES:
        current_power = p
        while current_power <= MAX_VAL:
            indices = np.arange(current_power, MAX_VAL + 1, current_power)
            sp[indices] *= p
            current_power *= p
            if current_power > MAX_VAL:
                break

    # Check time
    if time.time() > deadline - 10:
        return 1000

    # Now we need to find n in [1000, N] that minimizes the number of distinct values
    # in sp[n+1], sp[n+2], ..., sp[n+24]
    
    # Strategy: Use a sliding window with a frequency dictionary
    # But we can optimize by noting that the values of sp are products of primes < 24
    # The number of distinct values in any window of 24 consecutive numbers is bounded
    
    # Let's use a more efficient approach:
    # 1. First, do a coarse scan with step size to find promising regions
    # 2. Then do a fine scan in those regions
    
    # Actually, let's just do the full scan but optimize the inner loop
    
    best_n = 1000
    best_distinct = float('inf')
    
    # Use a dictionary for frequency counts
    freq = {}
    
    # Initialize window for n=1000
    distinct_count = 0
    for i in range(1, WINDOW + 1):
        v = sp[1000 + i]
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct_count += 1
    
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Slide the window
    # We'll batch the time checks to reduce overhead
    BATCH_SIZE = 10000
    for n in range(1001, N + 1):
        # Remove sp[n] (which was sp[(n-1)+1])
        old_val = sp[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1
        
        # Add sp[n+24]
        new_val = sp[n + 24]
        if new_val in freq:
            freq[new_val] += 1
        else:
            freq[new_val] = 1
            distinct_count += 1
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
        
        # Periodic time check
        if (n - 1001) % BATCH_SIZE == 0 and time.time() > deadline - 5:
            break
    
    return best_n

# EVOLVE-BLOCK-END
