import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def extract_small_part(x):
        """Extract the part of x composed only of primes < 24."""
        for p in small_primes:
            while x % p == 0:
                x //= p
        return x  # Wait, this removes small primes. I want to KEEP small primes.
    
    # Actually I need to extract the factor composed of primes < 24,
    # i.e., remove all prime factors >= 24.
    def get_24_smooth_part(x):
        """Return the part of x with only prime factors < 24."""
        result = 1
        for p in small_primes:
            while x % p == 0:
                result *= p
                x //= p
        return result
    
    best_n = 1000
    best_count = float('inf')
    
    # Precompute for efficiency: we can't precompute all, but we can vectorize
    # Let's just iterate and check
    
    # First, let's try a quick scan to find good candidates
    # The minimum possible distinct count is 1
    
    # Let's do a linear scan but with early termination
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        parts = set()
        for i in range(1, 25):
            parts.add(get_24_smooth_part(n + i))
        
        count = len(parts)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
