import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    max_n = 1000000
    min_n = 10000
    
    # Precompute primes up to sqrt(max_n) for trial division
    limit = int(math.isqrt(max_n)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def largest_prime_factor(x, primes):
        """Compute largest prime factor of x using trial division by primes."""
        if x <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                lpf = p
                while x % p == 0:
                    x //= p
        if x > 1:
            lpf = x
        return lpf
    
    # Strategy: Try increasing bounds B on the largest prime factor
    # For each B, check if there exist consecutive n, n+1 in range where both are B-smooth
    # A number is B-smooth if all its prime factors are <= B
    
    def is_smooth(x, B, primes):
        """Check if x is B-smooth (all prime factors <= B)."""
        if x <= 1:
            return True
        for p in primes:
            if p > B:
                break
            if p * p > x:
                break
            while x % p == 0:
                x //= p
        return x <= B
    
    best_n = min_n
    best_val = float('inf')
    
    # First pass: quick scan with the current approach to get a baseline
    # Use a faster LPF computation for the initial scan
    # We'll use the small primes list for trial division
    
    # Actually, let's just do a direct search but be smarter about it
    # We want to minimize max(lpf(n), lpf(n+1))
    # Let's try to find pairs where both are very smooth
    
    # Precompute which numbers in range are "very smooth" (all prime factors small)
    # We'll check for increasing B values
    
    for B in [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        
        # Find all n in [min_n, max_n] where both n and n+1 are B-smooth
        found = False
        for n in range(min_n, max_n + 1):
            if time.time() > deadline:
                break
            
            if is_smooth(n, B, small_primes) and is_smooth(n + 1, B, small_primes):
                # Verify with exact LPF
                l1 = largest_prime_factor(n, small_primes)
                l2 = largest_prime_factor(n + 1, small_primes)
                val = max(l1, l2)
                if val < best_val:
                    best_val = val
                    best_n = n
                found = True
        
        if found:
            # If we found any pair with this B, we can stop trying larger B
            # because we want to minimize the largest prime factor
            break
    
    # If no smooth pair was found (shouldn't happen for reasonable B), do a full scan
    if best_val == float('inf'):
        for n in range(min_n, max_n + 1):
            if time.time() > deadline:
                break
            l1 = largest_prime_factor(n, small_primes)
            l2 = largest_prime_factor(n + 1, small_primes)
            val = max(l1, l2)
            if val < best_val:
                best_val = val
                best_n = n
    
    return best_n
