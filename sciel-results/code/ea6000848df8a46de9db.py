"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    def sieve_primes(limit):
        """Generate primes up to limit using simple sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def largest_prime_factor(n, primes):
        """Find the largest prime factor of n using trial division by given primes."""
        if n <= 1:
            return 1
        original = n
        lp = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                lp = p
        if n > 1:
            lp = n
        return lp
    
    # Generate primes up to 1000 for trial division
    primes = sieve_primes(1000)
    
    # Also generate larger primes for factoring the remainder
    # We need primes up to sqrt(1000000*1000001) ~ 31623
    large_primes = sieve_primes(31624)
    
    best_n = 10000
    best_lpf = float('inf')
    
    # First, do a quick scan to find a good initial candidate
    # Look for numbers where both n and n+1 are smooth (small prime factors)
    
    # Strategy: scan through range, for each n compute largest prime factor of n*(n+1)
    # But we can optimize: largest_prime_factor(n*(n+1)) = max(lpf(n), lpf(n+1))
    # because n and n+1 are coprime
    
    # To speed up, we can precompute lpf for a window
    
    # Let's do a segmented approach:
    # For each candidate n, we need lpf(n) and lpf(n+1)
    # Since n and n+1 are coprime, lpf(n*(n+1)) = max(lpf(n), lpf(n+1))
    
    # Precompute smallest prime factor (SPF) for numbers up to some limit
    # But numbers go up to 1000000, so let's use trial division with our prime list
    
    # For efficiency, let's scan in chunks and track best
    
    # Actually, let's think about this differently.
    # We want to minimize max(lpf(n), lpf(n+1))
    # This is minimized when both n and n+1 have only small prime factors
    
    # Let's do a direct scan but with early termination
    # For each n from 10000 to 1000000:
    #   Compute lpf(n) and lpf(n+1)
    #   Track max
    #   Update best if improved
    
    # To make this fast, we can use the fact that if at any point during trial division
    # of n we find a prime factor larger than current best_lpf, we can stop early
    
    def lpf_with_limit(n, primes, limit):
        """Find largest prime factor of n, but stop if it exceeds limit."""
        if n <= 1:
            return 1
        original = n
        lp = 1
        for p in primes:
            if p * p > n:
                break
            if p > limit:
                # If we've already exceeded our limit, no need to continue
                return limit + 1
            if n % p == 0:
                while n % p == 0:
                    n //= p
                lp = p
        if n > 1:
            if n > limit:
                return n
            lp = n
        return lp
    
    # Scan the range
    # Use a two-pointer approach: for each n, compute lpf(n) and lpf(n+1)
    # We can cache lpf values
    
    # Let's precompute lpf for all numbers in a window
    # Actually, for 10000 to 1000000, that's 990001 numbers
    # Each trial division takes at most ~168 primes (primes up to 1000)
    # So ~168 million operations, which might be too slow in Python
    
    # Better approach: use a sieve-like method to compute lpf for all numbers
    
    def compute_lpf_array(n_max, primes):
        """Compute largest prime factor for all numbers from 1 to n_max."""
        lpf = [1] * (n_max + 1)
        for p in primes:
            for multiple in range(p, n_max + 1, p):
                # p is a prime factor of multiple
                # The largest prime factor so far
                if p > lpf[multiple]:
                    lpf[multiple] = p
        # For numbers that are prime and > max(primes), lpf should be the number itself
        # But we only have primes up to 1000, so numbers > 1000 that are prime won't be caught
        # We need to handle this
        
        # Actually, the sieve above only marks multiples of primes in our list
        # A number like 99991 (prime) won't have any multiple marked by our primes
        # So lpf[99991] will remain 1, which is wrong
        
        # We need a different approach. Let's use SPF (smallest prime factor) sieve
        return lpf
    
    # Let's use a proper SPF sieve
    def compute_lpf_via_spf(n_max):
        """Compute largest prime factor for all numbers from 1 to n_max using SPF."""
        if n_max < 2:
            return [1] * (n_max + 1)
        
        spf = list(range(n_max + 1))  # spf[i] = smallest prime factor of i
        for i in range(2, int(math.isqrt(n_max)) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i*i, n_max + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        lpf = [1] * (n_max + 1)
        for i in range(2, n_max + 1):
            # Factorize i using spf
            x = i
            max_pf = 1
            while x > 1:
                p = spf[x]
                max_pf = max(max_pf, p)
                while x % p == 0:
                    x //= p
            lpf[i] = max_pf
        return lpf
    
    # Compute lpf for all numbers up to 1000001
    # This should be fast with the SPF approach
    n_max = 1000001
    lpf = compute_lpf_via_spf(n_max)
    
    # Now scan for the best n
    for n in range(10000, 1000000):
        current_max = max(lpf[n], lpf[n+1])
        if current_max < best_lpf:
            best_lpf = current_max
            best_n = n
        
        # Time check
        if (n - 10000) % 100000 == 0 and n > 10000:
            if time.time() - start_time > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
