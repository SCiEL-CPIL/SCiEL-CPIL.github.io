# EVOLVE-BLOCK-START
import numpy as np

def construct():
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size + 10 # Safety margin
    
    # 1. Sieve of Eratosthenes to find primes up to max_val
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Primes >= 24
    primes_large = primes[primes >= 24]
    
    # 2. Compute f[m] = part of m composed of primes < 24
    # f[m] = m / (product of p^k for p >= 24)
    # Initialize f[m] = m
    f = np.arange(max_val + 1, dtype=np.int64)
    
    # For each large prime p, divide f[m] by p as many times as possible
    for p in primes_large:
        # We need to divide f[m] by p for all m divisible by p
        # This is tricky to vectorize efficiently with "while" loops.
        # Alternative: Compute the "large part" L[m] and then f[m] = m // L[m]
        # L[m] starts at 1.
        # For each p, L[m] *= p for all m divisible by p^k.
        
        pass # We'll use the L[m] approach below
    
    # Reset f and use L approach
    L = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes_large:
        # For each power p^k, multiply L[m] by p for multiples of p^k
        power = p
        while power <= max_val:
            multiples = np.arange(power, max_val + 1, power)
            L[multiples] *= p
            power *= p
    
    f = np.arange(max_val + 1, dtype=np.int64) // L
    
    # 3. Sliding window to minimize distinct count of f[n+1]...f[n+24]
    # freq array for values in f
    # Max value in f is max_val
    freq = np.zeros(max_val + 1, dtype=np.int32)
    
    # Initialize window for n=1000 (indices 1001 to 1024)
    current_distinct = 0
    for i in range(1001, 1001 + window_size):
        v = f[i]
        if freq[v] == 0:
            current_distinct += 1
        freq[v] += 1
        
    best_n = 1000
    best_count = current_distinct
    
    # Slide window
    for n in range(1001, max_n + 1):
        # Remove f[n]
        v_out = f[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            current_distinct -= 1
            
        # Add f[n+24]
        v_in = f[n + 24]
        if freq[v_in] == 0:
            current_distinct += 1
        freq[v_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break # Can't do better than 1 distinct factor value?
                # Actually, can it be 1?
                # If f[n+1] == ... == f[n+24], then yes.
                # This requires all numbers in the window to have the same small part.
                
    return best_n

# EVOLVE-BLOCK-END
