"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)  # 9
    
    # For each number in [1001, 1000024], compute bitmask of which small primes divide it
    # We need n in [1000, 1000000], so n+i in [1001, 1000024]
    lo = 1001
    hi = 1000024  # inclusive
    size = hi - lo + 1  # 1000024 - 1001 + 1 = 999024
    
    # Precompute bitmask for each number
    masks = [0] * size
    for pi, p in enumerate(primes):
        bit = 1 << pi
        # For each multiple of p in [lo, hi], set the bit
        first_mult = ((lo + p - 1) // p) * p
        for m in range(first_mult, hi + 1, p):
            idx = m - lo
            masks[idx] |= bit
    
    # Now slide a window of 24 over masks[0..size-1]
    # Window starting at index i covers n = lo - 1 + i, valid if n in [1000, 1000000]
    # i.e., lo - 1 + i >= 1000 => i >= 0 (since lo=1001, lo-1=1000)
    # lo - 1 + i <= 1000000 => i <= 1000000 - 1000 = 999000
    # Window covers indices i, i+1, ..., i+23, so i+23 < size => i <= size - 24
    
    max_i = min(999000, size - 24)
    
    best_count = num_primes + 1  # worst case is all 9 primes present
    best_n = 1000
    
    # Sliding window: maintain count of how many times each prime bit appears in window
    bit_counts = [0] * num_primes
    
    # Initialize window at i=0
    for j in range(24):
        m = masks[j]
        for pi in range(num_primes):
            if m & (1 << pi):
                bit_counts[pi] += 1
    
    # Count distinct primes in initial window
    distinct = sum(1 for c in bit_counts if c > 0)
    if distinct < best_count:
        best_count = distinct
        best_n = lo - 1  # n = 1000
    
    # Slide window
    for i in range(1, max_i + 1):
        # Remove element at i-1
        m_out = masks[i - 1]
        for pi in range(num_primes):
            if m_out & (1 << pi):
                bit_counts[pi] -= 1
        
        # Add element at i+23
        m_in = masks[i + 23]
        for pi in range(num_primes):
            if m_in & (1 << pi):
                bit_counts[pi] += 1
        
        distinct = sum(1 for c in bit_counts if c > 0)
        if distinct < best_count:
            best_count = distinct
            best_n = lo - 1 + i
            if best_count <= 2:
                # Very good, hard to beat
                break
        
        # Time check
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
