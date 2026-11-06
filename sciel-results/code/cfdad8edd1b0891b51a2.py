import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1 # We need to check n+1 up to 1000001

    # Precompute primes up to a reasonable limit for smooth number generation
    # If the optimal K is small, we won't need many primes.
    # Let's sieve primes up to 200.
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(200)
    
    # Function to generate all y-smooth numbers <= limit
    def generate_smooth_numbers(y, limit):
        # y is the largest prime allowed
        # Find primes <= y
        allowed_primes = [p for p in primes if p <= y]
        if not allowed_primes:
            return set()
            
        smooth_nums = {1}
        # Recursive generation or iterative expansion
        # Iterative expansion is easier but might be slow if not careful
        # Let's use a priority queue or sorted list to generate in order?
        # Actually, for a set, we can just expand.
        # Start with 1.
        # For each prime p in allowed_primes:
        #   For each existing num, multiply by p, p^2, ... until > limit
        
        current_smooth = [1]
        for p in allowed_primes:
            new_nums = []
            for num in current_smooth:
                val = num
                while val * p <= limit:
                    val *= p
                    new_nums.append(val)
            # Add new_nums to the set
            for num in new_nums:
                smooth_nums.add(num)
            # Update current_smooth to include the new numbers?
            # No, we need to combine primes.
            # The above logic is flawed. It only generates numbers with a single prime factor if not careful.
            # We need to generate all combinations.
            
        # Correct generation:
        smooth_set = {1}
        # We can use a recursive helper
        def gen(idx, current_val):
            if idx == len(allowed_primes):
                if current_val <= limit:
                    smooth_set.add(current_val)
                return
            p = allowed_primes[idx]
            # Try all powers of p
            val = current_val
            while val <= limit:
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        return smooth_set

    # Try increasing K
    # K is the largest prime factor allowed.
    # We iterate through primes as potential K.
    # If the optimal K is not a prime, it doesn't matter, we just need the set of primes <= K.
    # So we can just iterate K = 2, 3, 4, ... or just primes.
    # Iterating primes is sufficient because adding a composite K adds no new primes.
    
    for K in primes:
        if time.time() - start_time > TIME_LIMIT:
            # Fallback to sieve
            break
            
        smooth_nums = generate_smooth_numbers(K, LIMIT)
        
        # Check for consecutive pairs n, n+1 in smooth_nums
        # n must be in [10000, 1000000]
        # n+1 must be in smooth_nums
        found = False
        # To make this fast, sort the smooth numbers and use a set for O(1) lookup
        # Or just iterate through the sorted list of smooth numbers in range
        sorted_smooth = sorted([x for x in smooth_nums if x >= N_MIN and x <= N_MAX])
        
        # Convert to set for O(1) lookup of n+1
        # We already have smooth_set, which is a set.
        
        for n in sorted_smooth:
            if (n + 1) in smooth_set:
                return int(n)
        
        # If not found, try next K

    # Fallback: Sieve method (same as original)
    # This ensures we always return a valid answer
    N_PLUS_1 = N_MAX + 1
    SIZE = N_PLUS_1 + 1
    lpf = np.zeros(SIZE, dtype=np.int32)
    lpf[2::2] = 2
    for i in range(3, SIZE, 2):
        if time.time() - start_time > TIME_LIMIT - 5:
            return 10000
        if lpf[i] == 0:
            lpf[i::i] = i
            
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_np1 = lpf[N_MIN+1:N_MAX+2]
    scores = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(scores)
    best_n = N_MIN + min_idx
    return int(best_n)
