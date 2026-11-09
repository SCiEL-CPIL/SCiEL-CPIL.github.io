"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

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
    # This is more efficient than the iterative division method
    for p in PRIMES:
        # Start with p^1
        current_power = p
        while current_power <= MAX_VAL:
            # For all m divisible by current_power, multiply sp[m] by p
            # We do this by iterating over multiples of current_power
            # sp[m] *= p for m in range(current_power, MAX_VAL+1, current_power)
            indices = np.arange(current_power, MAX_VAL + 1, current_power)
            sp[indices] *= p
            current_power *= p
            if current_power > MAX_VAL:
                break

    # Sliding window over n in [1000, N], window is sp[n+1], sp[n+2], ..., sp[n+24]
    # For each window, count distinct values, minimize count
    best_n = 1000
    best_distinct = float('inf')

    # Use sliding window with a dictionary for frequency counts
    # Initial window: n=1000, values sp[1001] to sp[1024]
    # We use a dict for O(1) updates
    freq = {}
    
    # Helper to update freq and return distinct count change
    def add_val(v):
        if v in freq:
            freq[v] += 1
            return 0
        else:
            freq[v] = 1
            return 1

    def remove_val(v):
        freq[v] -= 1
        if freq[v] == 0:
            del freq[v]
            return -1
        return 0

    distinct_count = 0
    # Initialize window for n=1000
    for i in range(1, WINDOW + 1):
        v = sp[1000 + i]
        distinct_count += add_val(v)
        
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000

    # Slide the window
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        
        # Remove sp[n] (which was sp[(n-1)+1])
        old_val = sp[n]
        distinct_count += remove_val(old_val)
        
        # Add sp[n+24]
        new_val = sp[n + 24]
        distinct_count += add_val(new_val)
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
