"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Range of n is [1000, 1000000]
    # We need to check n+1 to n+24
    # Max value to check: 1000000 + 24 = 1000024
    max_val = 1000024
    
    # Precompute masks for each number
    # mask[x] is a bitfield where bit i is set if primes[i] divides x
    # Note: We only care about primes < 24.
    # If x is divisible by 2, bit 0 is set.
    # The "extracted factor" contains a prime p if p divides x.
    # So the set of distinct extracted factors' primes is the union of primes dividing any n+i.
    
    masks = [0] * (max_val + 1)
    
    # Sieve-like assignment of masks
    # For each prime p, mark multiples of p with the corresponding bit
    for i, p in enumerate(primes):
        bit = 1 << i
        for j in range(p, max_val + 1, p):
            masks[j] |= bit
            
    best_n = 1000
    best_count = float('inf')
    
    # Iterate through valid n
    # We can optimize by breaking early if we find count 1 (theoretical min > 0 likely)
    # But let's find the global min.
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
            
        union_mask = 0
        # Sum masks for n+1 to n+24
        # Using a loop for clarity, can be unrolled or optimized if needed
        for i in range(1, 25):
            union_mask |= masks[n + i]
            
        # Count bits
        # bin(x).count('1') is fast enough for 1e6 iterations
        count = bin(union_mask).count('1')
        
        if count < best_count:
            best_count = count
            best_n = n
            # If we find 1, it's likely the minimum (0 is impossible/extremely rare)
            # If we find 0, definitely min.
            if best_count <= 1:
                # Can't get better than 1 (unless 0)
                # We could continue to check for 0, but 1 is a strong candidate for min.
                # Let's check if 0 is possible.
                # If best_count == 0, break.
                if best_count == 0:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
